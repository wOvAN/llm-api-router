module llm-api-router/sdktest

go 1.27

require (
	github.com/anthropics/anthropic-sdk-go v1.66.0
	github.com/openai/openai-go/v3 v3.52.0
	llm-api-router v0.0.0
)

require (
	github.com/bahlo/generic-list-go v0.2.0 // indirect
	github.com/buger/jsonparser v1.1.2 // indirect
	github.com/invopop/jsonschema v0.14.0 // indirect
	github.com/pb33f/ordered-map/v2 v2.3.1 // indirect
	github.com/standard-webhooks/standard-webhooks/libraries v0.0.1 // indirect
	github.com/tidwall/gjson v1.19.0 // indirect
	github.com/tidwall/match v1.1.1 // indirect
	github.com/tidwall/pretty v1.2.1 // indirect
	github.com/tidwall/sjson v1.2.5 // indirect
	go.yaml.in/yaml/v4 v4.0.0-rc.2 // indirect
	golang.org/x/sync v0.16.0 // indirect
)

// The SDK sources come from the .info/ reference clones (gitignored, not part
// of the build), and the router under test is the parent module.
replace (

	github.com/anthropics/anthropic-sdk-go => ../.info/anthropic-sdk-go

	github.com/openai/openai-go/v3 => ../.info/openai-go
	llm-api-router => ..
)
