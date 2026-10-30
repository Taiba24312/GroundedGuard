"""Turn a RAGTruth `source_info` into the context string the model reads.

Rules (decided in Module 1 analysis):
  QA       -> question first, then the passages (dict keys: question, passages)
  Summary  -> the article text as-is (source_info is a string)
  Data2txt -> compact "key: value" lines instead of Python's str(dict).
              None / False values are KEPT: `due_to_null` hallucinations depend on them.

The official baseline uses f"{source_info}" for Data2txt. Our format differs only in
how the evidence is written; the labels and the metric live on the response, so
results stay comparable.
"""


def _fmt(value) -> str:
    """Compact, readable rendering of a nested value."""
    if isinstance(value, dict):
        return "(" + "; ".join(f"{k}: {_fmt(v)}" for k, v in value.items()) + ")"
    if isinstance(value, list):
        return "[" + " | ".join(_fmt(v) for v in value) + "]"
    return str(value)  # str / int / float / bool / None -> "None", "True", "False"


def serialize_data2txt(info: dict) -> str:
    lines = []
    for key, value in info.items():
        if isinstance(value, dict):
            inner = "; ".join(f"{k}: {_fmt(v)}" for k, v in value.items())
            lines.append(f"{key}: {inner}")
        elif isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            # e.g. review_info: one line per review
            lines.append(f"{key}:")
            for i, item in enumerate(value, 1):
                inner = "; ".join(f"{k}: {_fmt(v)}" for k, v in item.items())
                lines.append(f"- {i}. {inner}")
        else:
            lines.append(f"{key}: {_fmt(value)}")
    return "\n".join(lines)


def context_parts(task_type: str, source_info) -> tuple[str, str]:
    """(prefix, body). The prefix is repeated at the start of every context window
    (QA: the question); only the body is split into windows."""
    if task_type == "QA":
        if not isinstance(source_info, dict) or not {"question", "passages"} <= source_info.keys():
            raise ValueError(f"QA source_info must be a dict with question/passages, got {type(source_info)}")
        return f"Question: {source_info['question'].strip()}", source_info["passages"].strip()
    if task_type == "Summary":
        if not isinstance(source_info, str):
            raise ValueError(f"Summary source_info must be str, got {type(source_info)}")
        return "", source_info.strip()
    if task_type == "Data2txt":
        if not isinstance(source_info, dict):
            raise ValueError(f"Data2txt source_info must be dict, got {type(source_info)}")
        return "", serialize_data2txt(source_info)
    raise ValueError(f"unknown task_type: {task_type!r}")


def join_context(prefix: str, body: str) -> str:
    return f"{prefix}\n\n{body}" if prefix else body


def serialize_context(task_type: str, source_info) -> str:
    return join_context(*context_parts(task_type, source_info))