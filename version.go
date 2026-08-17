package main

// Version is the build version, injected at compile time via
// -ldflags "-X main.Version=...". Local builds report "dev".
var Version = "dev"
