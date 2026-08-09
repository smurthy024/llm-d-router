package loader

import (
	"strings"
	"testing"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol"
	_ "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dprank"
)

const prodCfg = `
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
  - type: inflight-load-producer
    parameters: {addEstimatedOutputTokens: true, outputRatio: 0.2, maxEstimatedOutputTokens: 1000}
  - type: token-load-scorer
    parameters: {queueThresholdTokens: 11264000}
  - type: max-score-picker
  - type: active-request-scorer
  - type: prefix-cache-scorer
  - type: kv-cache-utilization-scorer
  - type: concurrency-detector
    parameters: {concurrencyMode: "hybrid", maxTokenConcurrency: 11264000, maxConcurrency: 512}
  - type: dp-rank-router
    parameters: {dpSize: 8, loadWeight: 1.0}
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: concurrency-detector
      - pluginRef: active-request-scorer
        weight: 4
      - pluginRef: token-load-scorer
        weight: 4
      - pluginRef: kv-cache-utilization-scorer
        weight: 2
      - pluginRef: prefix-cache-scorer
        weight: 3
      - pluginRef: max-score-picker
flowControl:
  saturationDetector: {pluginRef: concurrency-detector}
  defaultRequestTTL: 300s
  defaultPriorityBand:
    maxBytes: 512Mi
    maxRequests: 1
    orderingPolicyRef: fcfs-ordering-policy
    fairnessPolicyRef: global-strict-fairness-policy
featureGates: [flowControl]
`

func init() { RegisterFeatureGate(flowcontrol.FeatureGate, false) }

func TestProdConfigWithDPRankLoads(t *testing.T) {
	if _, _, err := LoadRawConfig([]byte(prodCfg), logr.Discard()); err != nil {
		t.Fatalf("production config + dp-rank-router failed to load: %v", err)
	}
}

func TestProdConfigWithoutDPRankStillLoads(t *testing.T) {
	// phase-1 rollout: new image, ConfigMap untouched.
	cfg := strings.Replace(prodCfg,
		"  - type: dp-rank-router\n    parameters: {dpSize: 8, loadWeight: 1.0}\n", "", 1)
	if _, _, err := LoadRawConfig([]byte(cfg), logr.Discard()); err != nil {
		t.Fatalf("existing config regressed on the new image: %v", err)
	}
}
