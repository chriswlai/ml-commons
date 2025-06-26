/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.agent.tracing;

import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.telemetry.tracing.Span;
import org.opensearch.telemetry.tracing.SpanContext;
import org.opensearch.telemetry.tracing.SpanCreationContext;
import org.opensearch.telemetry.tracing.Tracer;
import org.opensearch.telemetry.tracing.attributes.Attributes;
import java.util.Map;

public abstract class AbstractMLTracer {
    protected final Tracer tracer;
    protected final MLFeatureEnabledSetting mlFeatureEnabledSetting;

    protected AbstractMLTracer(Tracer tracer, MLFeatureEnabledSetting mlFeatureEnabledSetting) {
        this.tracer = tracer;
        this.mlFeatureEnabledSetting = mlFeatureEnabledSetting;
    }

    public abstract Span startSpan(String name, Map<String, String> attributes, Span parentSpan);

    public abstract void endSpan(Span span);
}