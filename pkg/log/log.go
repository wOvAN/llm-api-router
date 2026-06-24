package log

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"
)

type Level = slog.Level

const (
	LevelDebug = slog.LevelDebug
	LevelInfo  = slog.LevelInfo
	LevelWarn  = slog.LevelWarn
	LevelError = slog.LevelError
)

type Logger struct {
	inner *slog.Logger
}

var defaultLogger = New(os.Stderr, LevelInfo)

func InitFromEnv() {
	raw := os.Getenv("LOG_LEVEL")
	if raw == "" {
		return
	}
	var lvl Level
	switch strings.ToLower(raw) {
	case "debug":
		lvl = LevelDebug
	case "info":
		lvl = LevelInfo
	case "warn":
		lvl = LevelWarn
	case "error":
		lvl = LevelError
	default:
		fmt.Fprintf(os.Stderr, "invalid LOG_LEVEL=%q, using info\n", raw)
		return
	}
	SetDefault(New(os.Stderr, lvl))
}

func New(w io.Writer, level Level) *Logger {
	return &Logger{
		inner: slog.New(slog.NewTextHandler(w, &slog.HandlerOptions{Level: level})),
	}
}

func SetDefault(l *Logger) {
	defaultLogger = l
}

func Default() *Logger {
	return defaultLogger
}

func (l *Logger) With(args ...any) *Logger {
	return &Logger{inner: l.inner.With(args...)}
}

func (l *Logger) Debug(msg string, args ...any) {
	l.inner.Debug(msg, args...)
}

func (l *Logger) Info(msg string, args ...any) {
	l.inner.Info(msg, args...)
}

func (l *Logger) Warn(msg string, args ...any) {
	l.inner.Warn(msg, args...)
}

func (l *Logger) Error(msg string, args ...any) {
	l.inner.Error(msg, args...)
}

func (l *Logger) Fatal(msg string, args ...any) {
	l.inner.Error(msg, args...)
	os.Exit(1)
}

func (l *Logger) Debugf(format string, args ...any) {
	l.inner.Debug(fmt.Sprintf(format, args...))
}

func (l *Logger) Infof(format string, args ...any) {
	l.inner.Info(fmt.Sprintf(format, args...))
}

func (l *Logger) Warnf(format string, args ...any) {
	l.inner.Warn(fmt.Sprintf(format, args...))
}

func (l *Logger) Errorf(format string, args ...any) {
	l.inner.Error(fmt.Sprintf(format, args...))
}

func (l *Logger) Fatalf(format string, args ...any) {
	l.inner.Error(fmt.Sprintf(format, args...))
	os.Exit(1)
}

// Package-level convenience functions.

func Debug(msg string, args ...any) { defaultLogger.Debug(msg, args...) }
func Info(msg string, args ...any)  { defaultLogger.Info(msg, args...) }
func Warn(msg string, args ...any)  { defaultLogger.Warn(msg, args...) }
func Error(msg string, args ...any) { defaultLogger.Error(msg, args...) }
func Fatal(msg string, args ...any) { defaultLogger.Fatal(msg, args...) }

func Debugf(format string, args ...any) { defaultLogger.Debugf(format, args...) }
func Infof(format string, args ...any)  { defaultLogger.Infof(format, args...) }
func Warnf(format string, args ...any)  { defaultLogger.Warnf(format, args...) }
func Errorf(format string, args ...any) { defaultLogger.Errorf(format, args...) }
func Fatalf(format string, args ...any) { defaultLogger.Fatalf(format, args...) }
