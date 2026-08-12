package agent

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// validate.go ports validateToolArguments from
// reference/pi/packages/ai/src/utils/validation.ts. The TS original uses
// TypeBox; this port implements the same behavior for plain JSON Schemas
// (the non-TypeBox path in the TS): primitive coercion followed by a
// recursive type/required/enum check. Unsupported keywords (format, pattern,
// min/max, etc.) are ignored rather than enforced.

type jsonSchema = map[string]any

// validateToolArguments validates tool call arguments against the tool's JSON
// Schema, returning the validated (and potentially coerced) arguments. It
// returns an error with the TS message format when validation fails.
func validateToolArguments(tool Tool, toolName string, args map[string]any) (map[string]any, error) {
	var schema jsonSchema
	if len(tool.Parameters) > 0 {
		if err := json.Unmarshal(tool.Parameters, &schema); err != nil {
			return nil, fmt.Errorf("Validation failed for tool %q: invalid parameters schema: %v", toolName, err)
		}
	}
	if schema == nil {
		// No schema: accept arguments as-is.
		if args == nil {
			return map[string]any{}, nil
		}
		return args, nil
	}

	candidate := deepCopyMap(args)
	if coerced, ok := coerceWithJSONSchema(candidate, schema).(map[string]any); ok {
		candidate = coerced
	}

	var errs []string
	checkValue(candidate, schema, "", &errs)
	if len(errs) == 0 {
		return candidate, nil
	}

	received, _ := json.MarshalIndent(args, "", "  ")
	sort.Strings(errs)
	return nil, fmt.Errorf("Validation failed for tool %q:\n%s\n\nReceived arguments:\n%s",
		toolName, "  - "+strings.Join(errs, "\n  - "), received)
}

func deepCopyMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = deepCopyValue(v)
	}
	return out
}

func deepCopyValue(v any) any {
	switch t := v.(type) {
	case map[string]any:
		return deepCopyMap(t)
	case []any:
		out := make([]any, len(t))
		for i, item := range t {
			out[i] = deepCopyValue(item)
		}
		return out
	default:
		return v
	}
}

// schemaTypes returns the declared JSON types of a schema (TS getSchemaTypes).
func schemaTypes(schema jsonSchema) []string {
	switch t := schema["type"].(type) {
	case string:
		return []string{t}
	case []any:
		var types []string
		for _, item := range t {
			if s, ok := item.(string); ok {
				types = append(types, s)
			}
		}
		return types
	default:
		return nil
	}
}

// matchesJSONType reports whether a value matches a JSON Schema type
// (TS matchesJsonType).
func matchesJSONType(value any, typ string) bool {
	switch typ {
	case "number":
		return isJSONNumber(value)
	case "integer":
		f, ok := toFloat(value)
		return ok && f == float64(int64(f))
	case "boolean":
		_, ok := value.(bool)
		return ok
	case "string":
		_, ok := value.(string)
		return ok
	case "null":
		return value == nil
	case "array":
		_, ok := value.([]any)
		return ok
	case "object":
		_, ok := value.(map[string]any)
		return ok
	default:
		return false
	}
}

func isJSONNumber(value any) bool {
	_, ok := toFloat(value)
	return ok
}

func toFloat(value any) (float64, bool) {
	switch n := value.(type) {
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case json.Number:
		f, err := n.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}

// coercePrimitiveByType coerces scalars toward a schema type
// (TS coercePrimitiveByType).
func coercePrimitiveByType(value any, typ string) any {
	switch typ {
	case "number":
		switch v := value.(type) {
		case nil:
			return float64(0)
		case string:
			if strings.TrimSpace(v) != "" {
				var f float64
				if _, err := fmt.Sscanf(strings.TrimSpace(v), "%g", &f); err == nil {
					return f
				}
			}
		case bool:
			if v {
				return float64(1)
			}
			return float64(0)
		}
		return value
	case "integer":
		switch v := value.(type) {
		case nil:
			return float64(0)
		case string:
			if strings.TrimSpace(v) != "" {
				var f float64
				if _, err := fmt.Sscanf(strings.TrimSpace(v), "%g", &f); err == nil && f == float64(int64(f)) {
					return f
				}
			}
		case bool:
			if v {
				return float64(1)
			}
			return float64(0)
		}
		return value
	case "boolean":
		switch v := value.(type) {
		case nil:
			return false
		case string:
			if v == "true" {
				return true
			}
			if v == "false" {
				return false
			}
		case float64:
			if v == 1 {
				return true
			}
			if v == 0 {
				return false
			}
		}
		return value
	case "string":
		switch v := value.(type) {
		case nil:
			return ""
		case float64:
			return fmt.Sprintf("%v", v)
		case bool:
			return fmt.Sprintf("%v", v)
		}
		return value
	case "null":
		if value == "" {
			return nil
		}
		if f, ok := toFloat(value); ok && f == 0 {
			return nil
		}
		if b, ok := value.(bool); ok && !b {
			return nil
		}
		return value
	default:
		return value
	}
}

// coerceWithJSONSchema coerces a value toward a schema, recursing into
// objects, arrays, and union/composition keywords (TS coerceWithJsonSchema).
func coerceWithJSONSchema(value any, schema jsonSchema) any {
	next := value

	if allOf, ok := schema["allOf"].([]any); ok {
		for _, nested := range allOf {
			if s, ok := nested.(map[string]any); ok {
				next = coerceWithJSONSchema(next, s)
			}
		}
	}
	if anyOf, ok := schema["anyOf"].([]any); ok {
		next = coerceWithUnionSchema(next, anyOf)
	}
	if oneOf, ok := schema["oneOf"].([]any); ok {
		next = coerceWithUnionSchema(next, oneOf)
	}

	types := schemaTypes(schema)
	matchesUnionMember := false
	if len(types) > 1 {
		for _, typ := range types {
			if matchesJSONType(next, typ) {
				matchesUnionMember = true
				break
			}
		}
	}
	if len(types) > 0 && !matchesUnionMember {
		for _, typ := range types {
			if candidate := coercePrimitiveByType(next, typ); !valuesEqual(candidate, next) {
				next = candidate
				break
			}
		}
	}

	if obj, ok := next.(map[string]any); ok && containsType(types, "object") {
		applySchemaObjectCoercion(obj, schema)
	}
	if arr, ok := next.([]any); ok && containsType(types, "array") {
		applySchemaArrayCoercion(arr, schema)
	}

	return next
}

func containsType(types []string, typ string) bool {
	for _, t := range types {
		if t == typ {
			return true
		}
	}
	return false
}

func valuesEqual(a, b any) bool {
	return fmt.Sprintf("%#v", a) == fmt.Sprintf("%#v", b)
}

// coerceWithUnionSchema returns the value unchanged when it already validates
// against a union member, otherwise the first successful coercion
// (TS coerceWithUnionSchema).
func coerceWithUnionSchema(value any, schemas []any) any {
	for _, raw := range schemas {
		if s, ok := raw.(map[string]any); ok {
			var errs []string
			checkValue(value, s, "", &errs)
			if len(errs) == 0 {
				return value
			}
		}
	}
	for _, raw := range schemas {
		if s, ok := raw.(map[string]any); ok {
			candidate := coerceWithJSONSchema(deepCopyValue(value), s)
			var errs []string
			checkValue(candidate, s, "", &errs)
			if len(errs) == 0 {
				return candidate
			}
		}
	}
	return value
}

// applySchemaObjectCoercion coerces object property values in place
// (TS applySchemaObjectCoercion).
func applySchemaObjectCoercion(value map[string]any, schema jsonSchema) {
	properties, _ := schema["properties"].(map[string]any)
	for key, rawProp := range properties {
		propValue, present := value[key]
		if !present {
			continue
		}
		if propSchema, ok := rawProp.(map[string]any); ok {
			value[key] = coerceWithJSONSchema(propValue, propSchema)
		}
	}
	if additional, ok := schema["additionalProperties"].(map[string]any); ok {
		for key, propValue := range value {
			if _, defined := properties[key]; defined {
				continue
			}
			value[key] = coerceWithJSONSchema(propValue, additional)
		}
	}
}

// applySchemaArrayCoercion coerces array items in place
// (TS applySchemaArrayCoercion).
func applySchemaArrayCoercion(value []any, schema jsonSchema) {
	switch items := schema["items"].(type) {
	case []any:
		for i := range value {
			if i >= len(items) {
				break
			}
			if itemSchema, ok := items[i].(map[string]any); ok {
				value[i] = coerceWithJSONSchema(value[i], itemSchema)
			}
		}
	case map[string]any:
		for i := range value {
			value[i] = coerceWithJSONSchema(value[i], items)
		}
	}
}

// checkValue recursively validates a value against a schema, appending
// "path: message" entries to errs. Unsupported keywords are ignored.
func checkValue(value any, schema jsonSchema, path string, errs *[]string) {
	location := path
	if location == "" {
		location = "root"
	}

	if enum, ok := schema["enum"].([]any); ok && len(enum) > 0 {
		matched := false
		for _, allowed := range enum {
			if valuesEqual(allowed, value) {
				matched = true
				break
			}
		}
		if !matched {
			*errs = append(*errs, fmt.Sprintf("%s: must be one of %v", location, enum))
			return
		}
	}

	if constValue, ok := schema["const"]; ok {
		if !valuesEqual(constValue, value) {
			*errs = append(*errs, fmt.Sprintf("%s: must be %v", location, constValue))
			return
		}
	}

	types := schemaTypes(schema)
	if len(types) > 0 {
		matched := false
		for _, typ := range types {
			if matchesJSONType(value, typ) {
				matched = true
				break
			}
		}
		if !matched {
			*errs = append(*errs, fmt.Sprintf("%s: expected %s", location, strings.Join(types, " or ")))
			return
		}
	}

	if obj, ok := value.(map[string]any); ok {
		if required, ok := schema["required"].([]any); ok {
			for _, rawName := range required {
				name, ok := rawName.(string)
				if !ok {
					continue
				}
				if _, present := obj[name]; !present {
					missing := name
					if path != "" {
						missing = path + "." + name
					}
					*errs = append(*errs, fmt.Sprintf("%s: required property missing", missing))
				}
			}
		}
		if properties, ok := schema["properties"].(map[string]any); ok {
			keys := make([]string, 0, len(properties))
			for key := range properties {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			for _, key := range keys {
				propValue, present := obj[key]
				if !present {
					continue
				}
				if propSchema, ok := properties[key].(map[string]any); ok {
					checkValue(propValue, propSchema, joinPath(path, key), errs)
				}
			}
		}
	}

	if arr, ok := value.([]any); ok {
		switch items := schema["items"].(type) {
		case map[string]any:
			for i, item := range arr {
				checkValue(item, items, fmt.Sprintf("%s[%d]", pathOrRoot(path), i), errs)
			}
		case []any:
			for i, item := range arr {
				if i >= len(items) {
					break
				}
				if itemSchema, ok := items[i].(map[string]any); ok {
					checkValue(item, itemSchema, fmt.Sprintf("%s[%d]", pathOrRoot(path), i), errs)
				}
			}
		}
	}

	for _, keyword := range []string{"allOf", "anyOf", "oneOf"} {
		if subs, ok := schema[keyword].([]any); ok {
			if keyword == "allOf" {
				for _, raw := range subs {
					if s, ok := raw.(map[string]any); ok {
						checkValue(value, s, path, errs)
					}
				}
				continue
			}
			matched := false
			for _, raw := range subs {
				if s, ok := raw.(map[string]any); ok {
					var subErrs []string
					checkValue(value, s, path, &subErrs)
					if len(subErrs) == 0 {
						matched = true
						break
					}
				}
			}
			if !matched && len(subs) > 0 {
				*errs = append(*errs, fmt.Sprintf("%s: does not match any %s schema", location, keyword))
			}
		}
	}
}

func joinPath(base, key string) string {
	if base == "" {
		return key
	}
	return base + "." + key
}

func pathOrRoot(path string) string {
	if path == "" {
		return "root"
	}
	return path
}
