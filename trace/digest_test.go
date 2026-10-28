package trace

import "testing"

func TestDigestOpenAIStream(t *testing.T) {
	raw := []byte("data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\"lo\",\"reasoning_content\":\"thinking\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n")
	d := DigestResponse(raw, "text/event-stream")
	if d.Text != "Hello thinking" {
		t.Errorf("text = %q, want the deltas joined (content then reasoning)", d.Text)
	}
	if d.FinishReason != "stop" {
		t.Errorf("finish_reason = %q, want stop", d.FinishReason)
	}
	if d.Events["data"] != 3 {
		t.Errorf("events = %v, want data:3 (the [DONE] sentinel is not a frame)", d.Events)
	}
}

func TestDigestAnthropicStream(t *testing.T) {
	raw := []byte("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n" +
		"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hm\"}}\n\n" +
		"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"max_tokens\"},\"usage\":{\"output_tokens\":2}}\n\n" +
		"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	d := DigestResponse(raw, "text/event-stream")
	if d.Text != "Hi hm" {
		t.Errorf("text = %q, want body and reasoning separated by a space", d.Text)
	}
	if d.FinishReason != "max_tokens" {
		t.Errorf("finish_reason = %q, want max_tokens", d.FinishReason)
	}
	if d.Events["content_block_delta"] != 2 || d.Events["message_stop"] != 1 {
		t.Errorf("events = %v, want content_block_delta:2 message_stop:1", d.Events)
	}
}

func TestDigestResponsesStream(t *testing.T) {
	raw := []byte("event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\"Once\"}\n\n" +
		"event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"delta\":\" upon\"}\n\n" +
		"event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n")
	d := DigestResponse(raw, "text/event-stream")
	if d.Text != "Once upon" {
		t.Errorf("text = %q, want \"Once upon\"", d.Text)
	}
	if d.FinishReason != "completed" {
		t.Errorf("finish_reason = %q, want completed", d.FinishReason)
	}
}

func TestDigestAudioStream(t *testing.T) {
	raw := []byte("event: speech.audio.delta\ndata: {\"type\":\"speech.audio.delta\",\"delta\":\"AAAA\"}\n\n" +
		"event: transcript.text.delta\ndata: {\"type\":\"transcript.text.delta\",\"delta\":\"hi\"}\n\n" +
		"event: transcript.text.done\ndata: {\"type\":\"transcript.text.done\"}\n\n")
	d := DigestResponse(raw, "text/event-stream")
	if d.Text != "hi" {
		t.Errorf("text = %q, want the transcript only", d.Text)
	}
	if d.AudioDeltas != 1 {
		t.Errorf("audio_deltas = %d, want 1 (base64 PCM counted, never stored)", d.AudioDeltas)
	}
}

func TestDigestJSON(t *testing.T) {
	d := DigestResponse([]byte(`{"choices":[{"message":{"content":"Hi there"},"finish_reason":"stop"}]}`), "application/json")
	if d.Text != "Hi there" || d.FinishReason != "stop" {
		t.Errorf("digest = %+v, want the message content and finish_reason", d)
	}
	if d.Events != nil {
		t.Errorf("events = %v, want none for a JSON response", d.Events)
	}

	d = DigestResponse([]byte(`{"role":"assistant","stop_reason":"end_turn","content":[{"type":"text","text":"A "},{"type":"text","text":"B"}]}`), "application/json")
	if d.Text != "A B" || d.FinishReason != "end_turn" {
		t.Errorf("digest = %+v, want Anthropic content blocks joined", d)
	}
}

func TestDigestBinary(t *testing.T) {
	d := DigestResponse([]byte{0x52, 0x49, 0x46, 0x46}, "audio/wav")
	if d.Text != "" || d.Events != nil {
		t.Errorf("digest = %+v, want nothing parsed out of a binary body", d)
	}
}

func TestDigestUnparseableFramesIgnored(t *testing.T) {
	d := DigestResponse([]byte("data: not json\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n"), "text/event-stream")
	if d.Text != "ok" {
		t.Errorf("text = %q, want the frame that did parse", d.Text)
	}
}
