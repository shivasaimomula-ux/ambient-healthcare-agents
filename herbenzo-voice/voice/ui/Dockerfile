# Builds NVIDIA's WebRTC UI at the pinned commit with Herbenzo config/branding, then serves it with Caddy (HTTPS).
FROM node:22-alpine AS build
RUN apk add --no-cache bash git python3
WORKDIR /ui
COPY setup_ui.sh config.ts ./
RUN ./setup_ui.sh && cd .build/webrtc_ui && npm run build

FROM caddy:2-alpine
COPY --from=build /ui/.build/webrtc_ui/dist /srv
COPY Caddyfile /etc/caddy/Caddyfile
EXPOSE 80 443
