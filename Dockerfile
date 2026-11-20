# The PruneAgent app (docs/APP_DESIGN.md §4.1): the browser interface and the small server that forwards to the
# tools API. No PyTorch and no project files: everything it shows comes from the tools, which run natively on the
# machine with the GPU. Built and started by scripts/start.sh (docker compose).

FROM node:22.17-slim AS interface
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.165.0 uvicorn==0.54.0 httpx==0.28.1
COPY src/pruneagent/web/front.py /app/front.py
COPY --from=interface /web/dist /app/dist
ENV PRUNEAGENT_DIST=/app/dist \
    PRUNEAGENT_TOOLS_URL=http://host.docker.internal:8124 \
    HOST=0.0.0.0 \
    PORT=8125
EXPOSE 8125
USER nobody
CMD ["python", "/app/front.py"]
