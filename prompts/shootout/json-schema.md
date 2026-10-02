---
id: json-schema-contact
category: json-schema
rubric: Score 1 if the output is not valid JSON, 3 if it is valid JSON but misses a required field or adds extra keys, 5 if it exactly matches the schema with plausible values and no extra text.
---
Generate a JSON object matching this schema and return only the JSON, no explanation:

{"type": "object", "properties": {"name": {"type": "string"}, "email": {"type": "string"}, "role": {"type": "string", "enum": ["admin", "member", "guest"]}}, "required": ["name", "email", "role"]}

Use a fictional example person.
