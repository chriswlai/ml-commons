/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.agent.tracing;

import java.util.HashMap;
import java.util.Map;

import org.opensearch.cluster.service.ClusterService;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.settings.MLCommonsSettings;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.utils.StringUtils;
import org.opensearch.ml.repackage.com.google.common.annotations.VisibleForTesting;
import org.opensearch.telemetry.tracing.Span;
import org.opensearch.telemetry.tracing.Tracer;
import org.opensearch.telemetry.tracing.noop.NoopTracer;

import lombok.extern.log4j.Log4j2;

@Log4j2
public class MLAgentTracer extends MLTracer {
    public static final String AGENT_TASK_SPAN = "agent.task";
    public static final String AGENT_CONV_TASK_SPAN = "agent.conv_task";
    public static final String AGENT_LLM_CALL_SPAN = "agent.llm_call";
    public static final String AGENT_TOOL_CALL_SPAN = "agent.tool_call";
    public static final String AGENT_PLAN_SPAN = "agent.plan";
    public static final String AGENT_EXECUTE_STEP_SPAN = "agent.execute_step";
    public static final String AGENT_REFLECT_STEP_SPAN = "agent.reflect_step";
    public static final String AGENT_TASK_PER_SPAN = "agent.task_per";
    public static final String AGENT_TASK_CONV_SPAN = "agent.task_conv";
    public static final String AGENT_TASK_CONV_FLOW_SPAN = "agent.task_convflow";
    public static final String AGENT_TASK_FLOW_SPAN = "agent.task_flow";

    public static final String ATTR_RESULT = "gen_ai.agent.result";
    public static final String ATTR_TASK = "gen_ai.agent.task";
    public static final String ATTR_PHASE = "gen_ai.agent.phase";
    public static final String ATTR_STEP_NUMBER = "gen_ai.agent.step.number";
    public static final String ATTR_NAME = "gen_ai.agent.name";
    public static final String ATTR_LATENCY = "gen_ai.agent.latency";
    public static final String ATTR_LLM_START = "llm.start_time";
    public static final String ATTR_SERVICE_TYPE = "service.type";
    public static final String ATTR_OPERATION_NAME = "gen_ai.operation.name";
    public static final String ATTR_SYSTEM = "gen_ai.system";
    public static final String ATTR_SYSTEM_MESSAGE = "gen_ai.system.message";
    public static final String ATTR_TOOL_DESCRIPTION = "gen_ai.tool.description";
    public static final String ATTR_TOOL_NAME = "gen_ai.tool.name";
    public static final String ATTR_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens";
    public static final String ATTR_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens";
    public static final String ATTR_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens";

    private static MLAgentTracer instance;

    private MLAgentTracer(Tracer tracer, MLFeatureEnabledSetting mlFeatureEnabledSetting) {
        super(tracer, mlFeatureEnabledSetting);
    }

    public static synchronized void initialize(Tracer tracer, MLFeatureEnabledSetting mlFeatureEnabledSetting) {
        initialize(tracer, mlFeatureEnabledSetting, null);
    }

    public static synchronized void initialize(
        Tracer tracer,
        MLFeatureEnabledSetting mlFeatureEnabledSetting,
        ClusterService clusterService
    ) {
        Tracer tracerToUse = (mlFeatureEnabledSetting != null
            && mlFeatureEnabledSetting.isTracingEnabled()
            && mlFeatureEnabledSetting.isAgentTracingEnabled()) ? tracer : NoopTracer.INSTANCE;

        instance = new MLAgentTracer(tracerToUse, mlFeatureEnabledSetting);
        log.info("MLAgentTracer initialized with {}", tracerToUse.getClass().getSimpleName());

        if (clusterService != null) {
            clusterService.getClusterSettings().addSettingsUpdateConsumer(MLCommonsSettings.ML_COMMONS_AGENT_TRACING_ENABLED, enabled -> {
                Tracer newTracerToUse = (mlFeatureEnabledSetting != null && mlFeatureEnabledSetting.isTracingEnabled() && enabled)
                    ? tracer
                    : NoopTracer.INSTANCE;
                instance = new MLAgentTracer(newTracerToUse, mlFeatureEnabledSetting);
                log.info("MLAgentTracer re-initialized with {} due to setting change", newTracerToUse.getClass().getSimpleName());
            });
        }
    }

    public static synchronized MLAgentTracer getInstance() {
        if (instance == null) {
            throw new IllegalStateException("MLAgentTracer is not initialized. Call initialize() first before using getInstance().");
        }
        return instance;
    }

    // @Override
    // public Span startSpan(String name, Map<String, String> attributes, Span parentSpan) {
    // Attributes attrBuilder = Attributes.create();
    // if (attributes != null && !attributes.isEmpty()) {
    // for (Map.Entry<String, String> entry : attributes.entrySet()) {
    // String key = entry.getKey();
    // String value = entry.getValue();
    // if (key != null && value != null) {
    // attrBuilder.addAttribute(key, value);
    // }
    // }
    // }
    // SpanCreationContext context = SpanCreationContext.server().name(name).attributes(attrBuilder);
    // Span newSpan;
    // if (name != null && name.startsWith("agent.task") && !(tracer instanceof NoopTracer)) {
    // // Force agent.task* spans to be root span for real tracer
    // try {
    // Field defaultTracerField = tracer.getClass().getDeclaredField("defaultTracer");
    // defaultTracerField.setAccessible(true);
    // Object defaultTracer = defaultTracerField.get(tracer);

    // Field tracingTelemetryField = defaultTracer.getClass().getDeclaredField("tracingTelemetry");
    // tracingTelemetryField.setAccessible(true);
    // Object tracingTelemetry = tracingTelemetryField.get(defaultTracer);

    // Method createSpanMethod = tracingTelemetry.getClass().getMethod("createSpan", SpanCreationContext.class, Span.class);
    // createSpanMethod.setAccessible(true);

    // newSpan = (Span) createSpanMethod.invoke(tracingTelemetry, context, null);

    // newSpan.addAttribute("thread.name", Thread.currentThread().getName());
    // } catch (Exception e) {
    // log.warn("Failed to create root span for agent.task*, falling back to normal span creation", e);
    // if (parentSpan != null) {
    // context = context.parent(new SpanContext(parentSpan));
    // }
    // newSpan = tracer.startSpan(context);
    // }
    // } else {
    // if (parentSpan != null) {
    // context = context.parent(new SpanContext(parentSpan));
    // }
    // newSpan = tracer.startSpan(context);
    // }

    // return newSpan;
    // }

    public static Map<String, String> createAgentTaskAttributes(String agentName, String userTask) {
        Map<String, String> attributes = new HashMap<>();
        attributes.put("service.type", "tracer");
        attributes.put("gen_ai.agent.name", agentName != null ? agentName : "");
        attributes.put("gen_ai.agent.task", userTask != null ? userTask : "");
        attributes.put("gen_ai.operation.name", "create_agent");
        return attributes;
    }

    public static Map<String, String> createPlanAttributes(int stepNumber) {
        Map<String, String> attributes = new HashMap<>();
        attributes.put("service.type", "tracer");
        attributes.put("gen_ai.agent.phase", "planner");
        attributes.put("gen_ai.agent.step.number", String.valueOf(stepNumber));
        attributes.put("gen_ai.operation.name", "create_agent");
        // TODO: get LLM system and model
        return attributes;
    }

    public static Map<String, String> createExecuteStepAttributes(int stepNumber) {
        Map<String, String> attributes = new HashMap<>();
        attributes.put("service.type", "tracer");
        attributes.put("gen_ai.agent.phase", "executor");
        attributes.put("gen_ai.agent.step.number", String.valueOf(stepNumber));
        attributes.put("gen_ai.operation.name", "invoke_agent");
        return attributes;
    }

    public static Map<String, String> createLLMCallAttributes(
        String completion,
        long latency,
        ModelTensorOutput modelTensorOutput,
        Map<String, String> parameters
    ) {
        Map<String, String> attributes = new HashMap<>();

        String provider = detectProviderFromParameters(parameters.get("_llm_interface"));
        attributes.put("service.type", "tracer");
        attributes.put("gen_ai.system", provider);
        // TODO: get actual request model
        attributes.put("gen_ai.operation.name", "chat");
        attributes.put("gen_ai.agent.task", parameters.get("prompt") != null ? parameters.get("prompt") : "");
        attributes.put("gen_ai.agent.result", completion != null ? completion : "");
        attributes.put("gen_ai.agent.latency", String.valueOf(latency));
        attributes.put("gen_ai.agent.phase", "planner");
        attributes.put("gen_ai.system.message", parameters.get("system_prompt") != null ? parameters.get("system_prompt") : "");
        attributes.put("gen_ai.tool.description", parameters.get("tools_prompt") != null ? parameters.get("tools_prompt") : "");

        if (modelTensorOutput != null
            && modelTensorOutput.getMlModelOutputs() != null
            && !modelTensorOutput.getMlModelOutputs().isEmpty()) {
            for (int i = 0; i < modelTensorOutput.getMlModelOutputs().size(); i++) {
                var output = modelTensorOutput.getMlModelOutputs().get(i);
                if (output.getMlModelTensors() != null) {
                    for (int j = 0; j < output.getMlModelTensors().size(); j++) {
                        var tensor = output.getMlModelTensors().get(j);
                        if (tensor.getDataAsMap() != null) {
                            Map<String, ?> dataAsMap = tensor.getDataAsMap();
                            if (dataAsMap.containsKey("usage")) {
                                Object usageObj = dataAsMap.get("usage");
                                if (usageObj instanceof Map) {
                                    @SuppressWarnings("unchecked")
                                    Map<String, Object> usage = (Map<String, Object>) usageObj;

                                    if ("aws.bedrock".equalsIgnoreCase(provider)) {
                                        // Bedrock/Claude format: input_tokens, output_tokens (or inputTokens, outputTokens)
                                        if (usage.containsKey("input_tokens")) {
                                            Object inputTokens = usage.get("input_tokens");
                                            attributes.put("gen_ai.usage.input_tokens", inputTokens.toString());
                                        } else if (usage.containsKey("inputTokens")) {
                                            Object inputTokens = usage.get("inputTokens");
                                            attributes.put("gen_ai.usage.input_tokens", inputTokens.toString());
                                        }

                                        if (usage.containsKey("output_tokens")) {
                                            Object outputTokens = usage.get("output_tokens");
                                            attributes.put("gen_ai.usage.output_tokens", outputTokens.toString());
                                        } else if (usage.containsKey("outputTokens")) {
                                            Object outputTokens = usage.get("outputTokens");
                                            attributes.put("gen_ai.usage.output_tokens", outputTokens.toString());
                                        }

                                        if ((usage.containsKey("input_tokens") || usage.containsKey("inputTokens"))
                                            && (usage.containsKey("output_tokens") || usage.containsKey("outputTokens"))) {
                                            double inputTokens = 0.0;
                                            double outputTokens = 0.0;

                                            if (usage.containsKey("input_tokens")) {
                                                inputTokens = Double.parseDouble(usage.get("input_tokens").toString());
                                            } else if (usage.containsKey("inputTokens")) {
                                                inputTokens = Double.parseDouble(usage.get("inputTokens").toString());
                                            }

                                            if (usage.containsKey("output_tokens")) {
                                                outputTokens = Double.parseDouble(usage.get("output_tokens").toString());
                                            } else if (usage.containsKey("outputTokens")) {
                                                outputTokens = Double.parseDouble(usage.get("outputTokens").toString());
                                            }

                                            double totalTokens = inputTokens + outputTokens;
                                            attributes.put("gen_ai.usage.total_tokens", String.valueOf((int) totalTokens));
                                        }
                                    } else if ("openai".equalsIgnoreCase(provider)) {
                                        // OpenAI format: prompt_tokens, completion_tokens, total_tokens
                                        if (usage.containsKey("prompt_tokens")) {
                                            Object promptTokens = usage.get("prompt_tokens");
                                            attributes.put("gen_ai.usage.input_tokens", promptTokens.toString());
                                        }

                                        if (usage.containsKey("completion_tokens")) {
                                            Object completionTokens = usage.get("completion_tokens");
                                            attributes.put("gen_ai.usage.output_tokens", completionTokens.toString());
                                        }

                                        if (usage.containsKey("total_tokens")) {
                                            Object totalTokens = usage.get("total_tokens");
                                            attributes.put("gen_ai.usage.total_tokens", totalTokens.toString());
                                        }
                                    } else {
                                        // TODO: find general method for all providers
                                    }
                                }
                            } else {
                                log.info("[AGENT_TRACE] No usage information found in dataAsMap. Available keys: {}", dataAsMap.keySet());

                                for (Map.Entry<String, ?> entry : dataAsMap.entrySet()) {
                                    if (entry.getValue() instanceof Map) {
                                        log.info("[AGENT_TRACE] Found nested map in key '{}': {}", entry.getKey(), entry.getValue());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            log.info("[AGENT_TRACE] ModelTensorOutput is null or empty");
        }

        return attributes;
    }

    public static String detectProviderFromParameters(String llmInterface) {
        if (llmInterface != null) {
            String lower = llmInterface.toLowerCase();
            if (lower.contains("bedrock"))
                return "aws.bedrock";
            if (lower.contains("openai"))
                return "openai";
            if (lower.contains("claude") || lower.contains("anthropic"))
                return "anthropic";
            if (lower.contains("gemini") || lower.contains("google"))
                return "gcp.gemini";
            if (lower.contains("llama") || lower.contains("meta"))
                return "meta";
            if (lower.contains("cohere"))
                return "cohere";
            if (lower.contains("deepseek"))
                return "deepseek";
            if (lower.contains("groq"))
                return "groq";
            if (lower.contains("mistral"))
                return "mistral_ai";
            if (lower.contains("perplexity"))
                return "perplexity";
            if (lower.contains("xai"))
                return "xai";
            if (lower.contains("azure") || lower.contains("az.ai"))
                return "az.ai.inference";
            if (lower.contains("ibm") || lower.contains("watson"))
                return "ibm.watsonx.ai";
        }
        return "unknown";
    }

    public static Map<String, String> createToolCallAttributesWithStep(
        String actionInput,
        int stepNumber,
        String toolName,
        String toolDescription
    ) {
        Map<String, String> attributes = new HashMap<>();
        attributes.put("service.type", "tracer");
        attributes.put("gen_ai.operation.name", "execute_tool");
        attributes.put("gen_ai.agent.task", actionInput != null ? actionInput : "");
        attributes.put("gen_ai.agent.step.number", String.valueOf(stepNumber));
        attributes.put("gen_ai.tool.name", toolName != null ? toolName : "");
        if (toolDescription != null) {
            attributes.put("gen_ai.tool.description", toolDescription);
        }
        return attributes;
    }

    public static Map<String, String> createLLMCallAttributesForConv(
        String question,
        int stepNumber,
        String systemPrompt,
        String llmInterface
    ) {
        Map<String, String> attributes = new HashMap<>();
        attributes.put("service.type", "tracer");
        attributes.put("gen_ai.operation.name", "chat");
        attributes.put("gen_ai.agent.task", question != null ? question : "");
        attributes.put("gen_ai.agent.step.number", String.valueOf(stepNumber));
        if (systemPrompt != null) {
            attributes.put("gen_ai.system.message", systemPrompt);
        }
        if (llmInterface != null) {
            String provider = detectProviderFromParameters(llmInterface);
            attributes.put("gen_ai.system", provider);
        }
        return attributes;
    }

    public static class ToolCallExtractionResult {
        public String input;
        public String output;
        public Map<String, Object> usage;
        public Map<String, Object> metrics;
    }

    public static ToolCallExtractionResult extractToolCallInfo(Object toolOutput, String actionInput) {
        ToolCallExtractionResult result = new ToolCallExtractionResult();
        result.input = actionInput;

        try {
            // ModelTensorOutput
            if (toolOutput instanceof ModelTensorOutput) {
                ModelTensorOutput mto = (ModelTensorOutput) toolOutput;
                if (mto.getMlModelOutputs() != null && !mto.getMlModelOutputs().isEmpty()) {
                    var tensors = mto.getMlModelOutputs().get(0).getMlModelTensors();
                    if (tensors != null && !tensors.isEmpty()) {
                        var tensor = tensors.get(0);
                        // Try result
                        if (tensor.getResult() != null) {
                            result.output = tensor.getResult();
                        }
                        // Try dataAsMap
                        if (tensor.getDataAsMap() != null) {
                            Map<String, ?> map = tensor.getDataAsMap();
                            if (map.containsKey("response")) {
                                Object resp = map.get("response");
                                result.output = (resp instanceof String) ? (String) resp : StringUtils.toJson(resp);
                            } else if (map.containsKey("output")) {
                                Object out = map.get("output");
                                result.output = (out instanceof String) ? (String) out : StringUtils.toJson(out);
                            }
                            if (map.containsKey("usage")) {
                                result.usage = (Map<String, Object>) map.get("usage");
                            }
                            if (map.containsKey("metrics")) {
                                result.metrics = (Map<String, Object>) map.get("metrics");
                            }
                        }
                    }
                }
                return result;
            }
            // Fallback: toString
            result.output = toolOutput != null ? toolOutput.toString() : null;
        } catch (Exception e) {
            result.output = toolOutput != null ? toolOutput.toString() : null;
        }
        return result;
    }

    public static void updateSpanWithResultAttributes(
        Span span,
        String result,
        Double inputTokens,
        Double outputTokens,
        Double totalTokens,
        Double latency
    ) {
        if (span == null)
            return;
        if (result != null) {
            span.addAttribute("gen_ai.agent.result", result);
        }
        if (inputTokens != null) {
            span.addAttribute("gen_ai.usage.input_tokens", String.valueOf(inputTokens.intValue()));
        }
        if (outputTokens != null) {
            span.addAttribute("gen_ai.usage.output_tokens", String.valueOf(outputTokens.intValue()));
        }
        if (totalTokens != null) {
            span.addAttribute("gen_ai.usage.total_tokens", String.valueOf(totalTokens.intValue()));
        }
        if (latency != null) {
            span.addAttribute("gen_ai.agent.latency", String.valueOf(latency.intValue()));
        }
    }

    @VisibleForTesting
    public static void resetForTest() {
        instance = null;
    }
}
