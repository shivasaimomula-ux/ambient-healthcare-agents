"""Voice smoke test: a headless WebRTC caller that speaks synthesized utterances into the voice controller.

It connects exactly like the browser UI (audio + video transceivers, SDP offer over /ws), plays Riva-TTS
generated speech as its "microphone", listens to the bot's audio, and prints the transcripts the UI would show.

Needs the agent (:8081) and voice controller (:7860) running, and NVIDIA_API_KEY in the environment.

    uv run python scripts/voice_smoke.py [ws://localhost:7860/ws]
"""

from __future__ import annotations

import asyncio
import audioop  # deprecated in 3.13; fine on the pinned 3.12
import fractions
import json
import os
import sys
import time

import riva.client
import websockets
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import AudioFrame

UTTERANCES = [
    "Yes, go ahead.",
    "It's for me.",
    "I've had burning acidity for about three weeks, and it gets worse after spicy food.",
    "Around six.",
]
TTS_FUNCTION_ID = "877104f7-e885-42b9-8de8-f6e4c6303969"
RATE = 48000
FRAME_SAMPLES = 960  # 20 ms
SPEECH_RMS = 300
BOT_TURN_SILENCE_S = 2.5


def synthesize(text: str) -> bytes:
    auth = riva.client.Auth(
        uri="grpc.nvcf.nvidia.com:443",
        use_ssl=True,
        metadata_args=[
            ["function-id", TTS_FUNCTION_ID],
            ["authorization", "Bearer " + os.environ["NVIDIA_API_KEY"]],
        ],
    )
    service = riva.client.SpeechSynthesisService(auth)
    return service.synthesize(
        text, voice_name="Magpie-Multilingual.EN-US.Mia", language_code="en-US", sample_rate_hz=RATE
    ).audio


class ScriptedMic(MediaStreamTrack):
    """Real-time paced audio track: silence, or queued speech followed by one second of silence."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self.buffer = bytearray()
        self.pts = 0
        self.started: float | None = None
        self.finished_at: float | None = None

    def say(self, pcm: bytes) -> None:
        self.finished_at = None
        self.buffer += pcm + b"\x00" * RATE * 2

    async def recv(self) -> AudioFrame:
        if self.started is None:
            self.started = time.time()
        await asyncio.sleep(max(0.0, self.started + self.pts / RATE - time.time()))
        size = FRAME_SAMPLES * 2
        chunk = bytes(self.buffer[:size])
        del self.buffer[:size]
        if chunk and not self.buffer:
            self.finished_at = time.time()
        frame = AudioFrame(format="s16", layout="mono", samples=FRAME_SAMPLES)
        frame.planes[0].update(chunk.ljust(size, b"\x00"))
        frame.sample_rate = RATE
        frame.pts = self.pts
        frame.time_base = fractions.Fraction(1, RATE)
        self.pts += FRAME_SAMPLES
        return frame


class BotEar:
    def __init__(self) -> None:
        self.last_voice = 0.0
        self.first_voice: float | None = None
        self.seconds = 0.0

    async def listen(self, track: MediaStreamTrack) -> None:
        while True:
            try:
                frame = await track.recv()
            except Exception:
                return
            if audioop.rms(frame.to_ndarray().tobytes(), 2) > SPEECH_RMS:
                now = time.time()
                self.last_voice = now
                self.seconds += frame.samples / frame.sample_rate
                if self.first_voice is None:
                    self.first_voice = now

    async def wait_turn(self, after: float, timeout: float = 90.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            await asyncio.sleep(0.25)
            if self.last_voice > after and time.time() - self.last_voice > BOT_TURN_SILENCE_S:
                return True
        return False


async def main(url: str) -> int:
    speech = [synthesize(u) for u in UTTERANCES]
    mic, ear = ScriptedMic(), BotEar()
    pc = RTCPeerConnection()
    pc.addTransceiver(mic, direction="sendrecv")
    pc.addTransceiver("video", direction="sendrecv")

    @pc.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        if track.kind == "audio":
            asyncio.ensure_future(ear.listen(track))

    await pc.setLocalDescription(await pc.createOffer())
    transcripts: dict[tuple[str, str], str] = {}
    failures = 0
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}))
        answer = json.loads(await ws.recv())
        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

        async def read_transcripts() -> None:
            async for raw in ws:
                try:
                    message = json.loads(raw)
                except ValueError:
                    continue
                if "actor" in message:
                    transcripts[(message["actor"], message["message_id"])] = message["text"]

        reader = asyncio.create_task(read_transcripts())
        heard = await ear.wait_turn(after=time.time())
        print(f"greeting heard={heard} ({ear.seconds:.1f}s of bot audio)")
        failures += not heard
        for text, pcm in zip(UTTERANCES, speech, strict=True):
            mic.say(pcm)
            while mic.finished_at is None:
                await asyncio.sleep(0.05)
            user_stopped = mic.finished_at - 1.0
            ear.first_voice = None
            replied = await ear.wait_turn(after=max(ear.last_voice, user_stopped))
            failures += not replied
            lag = f"{ear.first_voice - user_stopped:.1f}s" if ear.first_voice else "n/a"
            print(f"user: {text!r} -> bot replied={replied}, first audio after {lag}")
        reader.cancel()

    print("\nTranscripts as the UI shows them:")
    for (actor, _), text in transcripts.items():
        print(f"  {actor:4}: {text[:150]}")
    await pc.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:7860/ws")))
