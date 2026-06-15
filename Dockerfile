# Build stage
FROM --platform=$BUILDPLATFORM golang:1.26-alpine AS builder

ARG TARGETARCH

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} go build -ldflags="-s -w" -o llm-api-router .

# Runtime stage
FROM alpine:3.21

RUN apk --no-cache add ca-certificates

WORKDIR /app

COPY --from=builder /app/llm-api-router .

EXPOSE 8080

ENV PORT=8080
ENV CONFIG_FILE=/app/config.json

CMD ["./llm-api-router"]
