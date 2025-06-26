/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.agent.tracing;

import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.telemetry.tracing.Tracer;
import org.opensearch.telemetry.tracing.Span;
import org.opensearch.telemetry.tracing.SpanContext;
import org.opensearch.telemetry.tracing.SpanCreationContext;
import org.opensearch.telemetry.tracing.attributes.Attributes;
import lombok.extern.log4j.Log4j2;
import java.util.Map;
// import java.util.concurrent.ConcurrentHashMap;
import org.opensearch.telemetry.tracing.noop.NoopTracer;

@Log4j2
public class MLAgentTracer extends AbstractMLTracer {
    private static MLAgentTracer instance;
    // private final Map<String, Span> operationSpans = new ConcurrentHashMap<>();

    private MLAgentTracer(Tracer tracer, MLFeatureEnabledSetting mlFeatureEnabledSetting) {
        super(tracer, mlFeatureEnabledSetting);
    }

    // public static synchronized void initialize(Tracer tracer, MLFeatureEnabledSetting mlFeatureEnabledSetting) {
    //     log.info("Initializing MLAgentTracer with tracer: {} and feature enabled setting: {}",
    //         tracer != null ? tracer.getClass().getSimpleName() : "null",
    //         mlFeatureEnabledSetting != null ? mlFeatureEnabledSetting.toString() : "null");
    //     instance = new MLAgentTracer(tracer, mlFeatureEnabledSetting);
    //     log.info("MLAgentTracer initialized successfully");
    // }

    public static synchronized void initialize(Tracer tracer, MLFeatureEnabledSetting mlFeatureEnabledSetting) {
        if (mlFeatureEnabledSetting == null || !mlFeatureEnabledSetting.isAgentTracingFeatureEnabled()) {
            // Static feature flag is off: do not initialize, do not trace at all
            instance = null;
            log.info("MLAgentTracer not initialized: agent tracing feature flag is disabled.");
            return;
        }
        Tracer tracerToUse = mlFeatureEnabledSetting.isAgentTracingEnabled() ? tracer : NoopTracer.INSTANCE;
        instance = new MLAgentTracer(tracerToUse, mlFeatureEnabledSetting);
        log.info("MLAgentTracer initialized with {}", tracerToUse.getClass().getSimpleName());
    }

    public static synchronized MLAgentTracer getInstance() {
        if (instance == null) {
            throw new IllegalStateException("MLAgentTracer is not initialized. Call initialize() first or check feature flag.");
        }
        return instance;
    }

    /**
     * Start a new agent span with the given name and attributes, and explicit parent.
     *
     * @param name The name of the span
     * @param attributes The attributes to add to the span
     * @param parentSpan The parent span, or null for root
     * @return The created span
     */
    @Override
    public Span startSpan(String name, Map<String, String> attributes, Span parentSpan) {
        if (tracer == null) {
            return null;
        }
        Attributes attrBuilder = Attributes.create();
        if (attributes != null && !attributes.isEmpty()) {
            for (Map.Entry<String, String> entry : attributes.entrySet()) {
                String key = entry.getKey();
                String value = entry.getValue();
                if (key != null && value != null) {
                    attrBuilder.addAttribute(key, value);
                }
            }
        }
        SpanCreationContext context = SpanCreationContext
            .server()
            .name(name)
            .attributes(attrBuilder);
        
        Span newSpan;
        if ("agent.task".equals(name)) {
            // For agent.task spans, bypass WrappedTracer's automatic parent detection
            // and directly call the underlying TracingTelemetry to create a true root span
            try {
                // Use reflection to access the underlying tracer through WrappedTracer
                java.lang.reflect.Field defaultTracerField = tracer.getClass().getDeclaredField("defaultTracer");
                defaultTracerField.setAccessible(true);
                Object defaultTracer = defaultTracerField.get(tracer);
                
                // Now get the TracingTelemetry from DefaultTracer
                java.lang.reflect.Field tracingTelemetryField = defaultTracer.getClass().getDeclaredField("tracingTelemetry");
                tracingTelemetryField.setAccessible(true);
                Object tracingTelemetry = tracingTelemetryField.get(defaultTracer);
                
                java.lang.reflect.Method createSpanMethod = tracingTelemetry.getClass()
                    .getMethod("createSpan", SpanCreationContext.class, Span.class);
                createSpanMethod.setAccessible(true);
                
                newSpan = (Span) createSpanMethod.invoke(tracingTelemetry, context, null);
                
                // Add default attributes that DefaultTracer would normally add
                newSpan.addAttribute("thread.name", Thread.currentThread().getName());
                
                log.debug("[AGENT_TRACE] Started ROOT agent span: {} | Thread: {}", 
                         name, Thread.currentThread().getName());
            } catch (Exception e) {
                log.warn("Failed to create root span for agent.task, falling back to normal span creation", e);
                // Fallback to normal span creation
                if (parentSpan != null) {
                    context = context.parent(new SpanContext(parentSpan));
                }
                newSpan = tracer.startSpan(context);
            }
        } else {
            // Normal span creation for all other spans
            if (parentSpan != null) {
                context = context.parent(new SpanContext(parentSpan));
            }
            newSpan = tracer.startSpan(context);
        }
        
        log.debug("[AGENT_TRACE] Started agent span: {} | Parent: {} | Thread: {}", 
                 name, 
                 parentSpan != null ? parentSpan.getSpanName() : "none",
                 Thread.currentThread().getName());
        return newSpan;
    }

    /**
     * End the current agent span.
     * 
     * @param span The span to end
     */
    @Override
    public void endSpan(Span span) {
        if (span == null || tracer == null) {
            return;
        }
        span.endSpan();
        log.debug("[AGENT_TRACE] Ended agent span: {} | Thread: {}", 
                 span.getSpanName(), 
                 Thread.currentThread().getName());
    }

    // /**
    //  * Store a span for a specific operation ID.
    //  * Useful for cross-thread span context propagation.
    //  * 
    //  * @param operationId The operation identifier
    //  * @param span The span to store
    //  */
    // public void storeSpan(String operationId, Span span) {
    //     if (operationId != null && span != null) {
    //         operationSpans.put(operationId, span);
    //     }
    // }

    // /**
    //  * Retrieve a stored span by operation ID.
    //  * 
    //  * @param operationId The operation identifier
    //  * @return The stored span, or null if not found
    //  */
    // public Span getStoredSpan(String operationId) {
    //     return operationId != null ? operationSpans.get(operationId) : null;
    // }

    // /**
    //  * Remove a stored span by operation ID.
    //  * 
    //  * @param operationId The operation identifier
    //  */
    // public void removeStoredSpan(String operationId) {
    //     if (operationId != null) {
    //         operationSpans.remove(operationId);
    //     }
    // }

    public Tracer getTracer() {
        return tracer;
    }
}