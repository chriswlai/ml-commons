package org.opensearch.ml.engine.algorithms.agent.tracing;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

import org.opensearch.commons.authuser.User;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.connector.MLCreateConnectorInput;
import org.opensearch.telemetry.tracing.Tracer;

public class MLConnectorTracer extends MLTracer {
    private static MLConnectorTracer instance;

    private MLConnectorTracer(Tracer tracer, MLFeatureEnabledSetting mlFeatureEnabledSetting) {
        super(tracer, mlFeatureEnabledSetting);
    }

    public static void initialize(Tracer tracer, MLFeatureEnabledSetting mlFeatureEnabledSetting) {
        instance = new MLConnectorTracer(tracer, mlFeatureEnabledSetting);
    }

    public static MLConnectorTracer getInstance() {
        if (instance == null) {
            throw new IllegalStateException("MLConnectorTracer is not initialized");
        }
        return instance;
    }

    public static Map<String, String> createConnectorAttributes(String connectorName, String connectorType, String tenantId) {
        Map<String, String> attributes = new HashMap<>();
        attributes.put("service.type", "tracer");
        attributes.put("ml.connector.name", connectorName);
        attributes.put("ml.connector.type", connectorType);
        attributes.put("ml.connector.tenant_id", tenantId);
        return attributes;
    }

    public static Map<String, String> createCreateAttributes(MLCreateConnectorInput input, User user) {
        Map<String, String> attributes = new HashMap<>();
        attributes.put("service.type", "tracer");
        attributes.put("ml.connector.name", input.getName());
        attributes.put("ml.connector.description", input.getDescription());
        attributes.put("ml.connector.version", input.getVersion());
        attributes.put("ml.connector.protocol", input.getProtocol());
        attributes.put("ml.connector.tenant_id", input.getTenantId());
        attributes.put("ml.connector.url", input.getUrl());
        attributes.put("ml.connector.headers", input.getHeaders() != null ? input.getHeaders().toString() : "");
        attributes.put("ml.connector.parameters", input.getParameters() != null ? input.getParameters().toString() : "");
        attributes.put("ml.connector.actions", input.getActions() != null ? input.getActions().toString() : "");
        attributes
            .put("ml.connector.client_config", input.getConnectorClientConfig() != null ? input.getConnectorClientConfig().toString() : "");
        attributes.put("ml.connector.backend_roles", input.getBackendRoles() != null ? String.join(",", input.getBackendRoles()) : "");
        attributes.put("ml.connector.add_all_backend_roles", String.valueOf(input.getAddAllBackendRoles()));
        attributes.put("ml.connector.access_mode", input.getAccess() != null ? input.getAccess().toString() : "");
        attributes.put("ml.connector.dry_run", String.valueOf(input.isDryRun()));
        attributes.put("ml.connector.update_connector", String.valueOf(input.isUpdateConnector()));
        if (user != null) {
            attributes.put("ml.user.name", user.getName());
            attributes.put("ml.user.backend_roles", user.getBackendRoles() != null ? String.join(",", user.getBackendRoles()) : "");
            attributes.put("ml.user.roles", user.getRoles() != null ? String.join(",", user.getRoles()) : "");
            attributes.put("ml.user.is_admin", String.valueOf(user.getRoles() != null && user.getRoles().contains("all_access")));
        }
        return attributes;
    }

    public static Map<String, String> createValidateAttributes(MLCreateConnectorInput input, User user, String result, String error) {
        Map<String, String> attributes = createCreateAttributes(input, user);
        attributes.put("ml.validation.result", result);
        if (error != null) {
            attributes.put("ml.validation.error", error);
        }
        return attributes;
    }

    public static Map<String, String> createIndexAttributes(
        MLCreateConnectorInput input,
        User user,
        String indexName,
        String result,
        String connectorId,
        Instant createdTime,
        Instant lastUpdatedTime,
        String error
    ) {
        Map<String, String> attributes = createCreateAttributes(input, user);
        attributes.put("ml.index.name", indexName);
        attributes.put("ml.index.result", result);
        if (connectorId != null) {
            attributes.put("ml.index.connector_id", connectorId);
        }
        if (createdTime != null) {
            attributes.put("ml.index.created_time", createdTime.toString());
        }
        if (lastUpdatedTime != null) {
            attributes.put("ml.index.last_updated_time", lastUpdatedTime.toString());
        }
        if (error != null) {
            attributes.put("ml.index.error", error);
        }
        return attributes;
    }
}
