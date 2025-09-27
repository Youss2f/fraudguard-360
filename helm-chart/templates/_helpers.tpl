# ==============================================================================
# FraudGuard 360 - Helm Template Helpers
# Common template functions for consistent labeling and naming
# ==============================================================================

{{/*
Expand the name of the chart.
*/}}
{{- define "fraudguard-360.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "fraudguard-360.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "fraudguard-360.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "fraudguard-360.labels" -}}
helm.sh/chart: {{ include "fraudguard-360.chart" . }}
{{ include "fraudguard-360.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: fraudguard-360
environment: {{ .Values.global.environment }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "fraudguard-360.selectorLabels" -}}
app.kubernetes.io/name: {{ include "fraudguard-360.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "fraudguard-360.serviceAccountName" -}}
{{- if .Values.security.serviceAccount.create }}
{{- default (include "fraudguard-360.fullname" .) .Values.security.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.security.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the namespace to use
*/}}
{{- define "fraudguard-360.namespace" -}}
{{- if .Values.namespace.create }}
{{- .Values.namespace.name }}
{{- else }}
{{- .Release.Namespace }}
{{- end }}
{{- end }}

{{/*
Common environment variables for database connectivity
*/}}
{{- define "fraudguard-360.database.env" -}}
- name: POSTGRES_HOST
  value: "{{ include "fraudguard-360.fullname" . }}-postgresql"
- name: POSTGRES_PORT
  value: "5432"
- name: POSTGRES_DB
  value: {{ .Values.postgresql.auth.database }}
- name: POSTGRES_USER
  value: {{ .Values.postgresql.auth.username }}
- name: POSTGRES_PASSWORD
  valueFrom:
    secretKeyRef:
      name: {{ .Values.postgresql.auth.existingSecret }}
      key: password
{{- end }}

{{/*
Common environment variables for Neo4j connectivity
*/}}
{{- define "fraudguard-360.neo4j.env" -}}
- name: NEO4J_URI
  value: "bolt://{{ include "fraudguard-360.fullname" . }}-neo4j:{{ .Values.neo4j.services.neo4j.ports.bolt }}"
- name: NEO4J_USER
  valueFrom:
    secretKeyRef:
      name: fraudguard-neo4j-secret
      key: username
- name: NEO4J_PASSWORD
  valueFrom:
    secretKeyRef:
      name: fraudguard-neo4j-secret
      key: password
{{- end }}

{{/*
Common environment variables for Kafka connectivity
*/}}
{{- define "fraudguard-360.kafka.env" -}}
- name: KAFKA_BOOTSTRAP_SERVERS
  value: "{{ include "fraudguard-360.fullname" . }}-kafka:9092"
- name: KAFKA_TOPIC_TRANSACTIONS
  value: "fraud-transactions"
- name: KAFKA_TOPIC_ALERTS
  value: "fraud-alerts"
- name: KAFKA_TOPIC_ANALYTICS
  value: "fraud-analytics"
{{- end }}

{{/*
Common security context
*/}}
{{- define "fraudguard-360.securityContext" -}}
securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities:
    drop:
    - ALL
  {{- if .Values.global.security.podSecurityContext.runAsNonRoot }}
  runAsNonRoot: {{ .Values.global.security.podSecurityContext.runAsNonRoot }}
  {{- end }}
  {{- if .Values.global.security.podSecurityContext.runAsUser }}
  runAsUser: {{ .Values.global.security.podSecurityContext.runAsUser }}
  {{- end }}
{{- end }}

{{/*
Common volume mounts for temporary directories
*/}}
{{- define "fraudguard-360.commonVolumeMounts" -}}
- name: temp
  mountPath: /tmp
- name: cache
  mountPath: /app/.cache
{{- end }}

{{/*
Common volumes for temporary directories
*/}}
{{- define "fraudguard-360.commonVolumes" -}}
- name: temp
  emptyDir: {}
- name: cache
  emptyDir: {}
{{- end }}

{{/*
Common resource limits based on environment
*/}}
{{- define "fraudguard-360.resources" -}}
{{- if eq .Values.global.environment "development" }}
resources:
  requests:
    cpu: {{ .Values.environments.development.resources.requests.cpu }}
    memory: {{ .Values.environments.development.resources.requests.memory }}
  limits:
    cpu: {{ .Values.environments.development.resources.limits.cpu }}
    memory: {{ .Values.environments.development.resources.limits.memory }}
{{- else }}
resources:
  {{- toYaml .resources | nindent 2 }}
{{- end }}
{{- end }}

{{/*
Monitoring annotations
*/}}
{{- define "fraudguard-360.monitoring.annotations" -}}
{{- if .Values.global.monitoring.enabled }}
prometheus.io/scrape: "true"
prometheus.io/path: "/metrics"
prometheus.io/port: "{{ .port | default 8000 }}"
{{- end }}
{{- end }}

{{/*
Create image pull secret names
*/}}
{{- define "fraudguard-360.imagePullSecrets" -}}
{{- if .Values.global.imagePullSecrets }}
imagePullSecrets:
{{- range .Values.global.imagePullSecrets }}
- name: {{ . }}
{{- end }}
{{- end }}
{{- end }}