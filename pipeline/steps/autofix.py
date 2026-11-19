"""BƯỚC 2/4 — Sửa tự động những lỗi bắt buộc mà máy sửa được AN TOÀN.   ★★

Test:  pytest tests/test_autofix.py -q

`fix(ds, issues, cfg) -> list[dict]`: duyệt `issues` (chỉ lỗi severity ERROR), sửa thẳng vào thư mục `ds`,
và trả về danh sách thay đổi; mỗi phần tử có khoá: image, line, action, before, after
(đây chính là file `03_fixes.csv` — bằng chứng pipeline đã làm gì, người có thể kiểm lại).

| code lỗi                                       | cách sửa                                    | action           |
|------------------------------------------------|---------------------------------------------|------------------|
| BAD_FORMAT, BAD_CLASS_ID, BOX_ZERO, DUP_BOX    | xoá dòng đó khỏi file nhãn                  | "drop_line"      |
| BOX_OUT_OF_IMAGE                               | cắt box về trong [0,1] (giữ nguyên lớp)     | "clip_box"       |
| MISSING_LABEL_FILE                             | KHÔNG sửa file (máy không thể bịa nhãn): ghi 1 dòng fix, before="(không có file)", after="chuyển người gán" | "needs_label" |
| BAD_CLASS_NAME                                 | ghi lại `names` trong data.yaml đúng `cfg["classes"]` | "rename_classes" |
| DUP_IMAGE_*, LEAK_TEST, CORRUPT_IMAGE, BAD_SIZE| xoá ảnh + nhãn khỏi `ds`                    | "remove_image"   |

Bẫy cần nghĩ tới (có test cho từng cái)
- Một file có NHIỀU dòng lỗi: số dòng trong `issues` là số dòng GỐC. Nếu xoá từng dòng một thì số dòng lệch.
  Gom theo file, rồi ghi lại MỘT lần.
- Sau khi cắt, box có thể không còn diện tích (nằm hẳn ngoài ảnh): lúc đó xoá dòng thay vì ghi box rỗng.
- Một dòng có 2 lỗi (vd. BOX_OUT_OF_IMAGE + DUP_BOX): xoá dòng thắng, chỉ ghi 1 fix `drop_line`, không ghi thêm `clip_box`. Dòng trống ở giữa file vẫn đếm vào số dòng gốc.
- Ảnh vừa bị xoá (DUP_IMAGE_*...) mà còn issue dòng của chính nó: bỏ qua êm, không được báo lỗi/chết.
- Không động tới lỗi WARN (cảnh báo là việc của người). Không sửa file ở `data/raw/` — chỉ sửa `ds`.
- Chạy lần 2 trên dữ liệu đã sửa phải không làm gì thêm (idempotent) — NGOẠI LỆ duy nhất: `needs_label` được ghi lại mỗi lần còn lỗi `MISSING_LABEL_FILE` (vì ảnh vẫn chưa có nhãn).

- `needs_label` KHÔNG tạo file nhãn rỗng (rỗng = "ảnh nền", khác với "chưa gán"). `review_queue` sẽ đẩy ảnh này lên đầu hàng đợi cho người.

Gợi ý: `yaml.safe_load / safe_dump`.
"""
from collections import defaultdict  # noqa: F401

import yaml  # noqa: F401

from ..common import ERROR, parse_yolo_line  # noqa: F401


def fix(ds, issues, cfg):
    drop_codes = {"BAD_FORMAT", "BAD_CLASS_ID", "BOX_ZERO", "DUP_BOX"}
    remove_codes = {"DUP_IMAGE_EXACT", "DUP_IMAGE_NEAR", "LEAK_TEST", "CORRUPT_IMAGE", "BAD_SIZE"}
    errors = [issue for issue in issues if issue.severity == ERROR]
    fixes = []

    remove_images = set()
    for issue in errors:
        if issue.code in remove_codes:
            remove_images.add(issue.image)
    for stem in sorted(remove_images):
        image_path = ds.image_path(stem)
        label_path = ds.label_path(stem)
        if image_path is not None:
            image_path.unlink(missing_ok=True)
        label_path.unlink(missing_ok=True)
        fixes.append({
            "image": stem,
            "line": "",
            "action": "remove_image",
            "before": "ảnh và nhãn hiện có",
            "after": "đã xoá",
        })

    needs_label = sorted({
        issue.image for issue in errors
        if issue.code == "MISSING_LABEL_FILE" and issue.image not in remove_images
    })
    for stem in needs_label:
        fixes.append({
            "image": stem,
            "line": "",
            "action": "needs_label",
            "before": "(không có file)",
            "after": "chuyển người gán",
        })

    by_image = defaultdict(list)
    for issue in errors:
        if issue.image not in remove_images and issue.line is not None:
            by_image[issue.image].append(issue)

    for stem, image_issues in by_image.items():
        label_path = ds.label_path(stem)
        if not label_path.exists():
            continue
        original_lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
        issues_by_line = defaultdict(list)
        for issue in image_issues:
            issues_by_line[issue.line].append(issue)

        changed = False
        output_lines = []
        for line_number, text in enumerate(original_lines, 1):
            line_issues = issues_by_line.get(line_number, [])
            if not line_issues:
                output_lines.append(text)
                continue

            drop_issue = next((issue for issue in line_issues if issue.code in drop_codes), None)
            if drop_issue is not None:
                fixes.append({
                    "image": stem,
                    "line": line_number,
                    "action": "drop_line",
                    "before": text,
                    "after": "",
                })
                changed = True
                continue

            if any(issue.code == "BOX_OUT_OF_IMAGE" for issue in line_issues):
                try:
                    class_id, cx, cy, width, height = parse_yolo_line(text)
                except ValueError:
                    output_lines.append(text)
                    continue
                left, top = max(0.0, cx - width / 2), max(0.0, cy - height / 2)
                right, bottom = min(1.0, cx + width / 2), min(1.0, cy + height / 2)
                clipped_width, clipped_height = right - left, bottom - top
                if clipped_width <= 0 or clipped_height <= 0:
                    fixes.append({
                        "image": stem,
                        "line": line_number,
                        "action": "drop_line",
                        "before": text,
                        "after": "",
                    })
                    changed = True
                    continue
                clipped = (
                    class_id,
                    (left + right) / 2,
                    (top + bottom) / 2,
                    clipped_width,
                    clipped_height,
                )
                after = " ".join(f"{value:.6g}" for value in clipped)
                output_lines.append(after)
                fixes.append({
                    "image": stem,
                    "line": line_number,
                    "action": "clip_box",
                    "before": text,
                    "after": after,
                })
                changed = True
                continue

            output_lines.append(text)

        if changed:
            label_path.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")

    if any(issue.code == "BAD_CLASS_NAME" and issue.image == "data.yaml" for issue in errors):
        if ds.yaml_path.exists():
            data = yaml.safe_load(ds.yaml_path.read_text(encoding="utf-8")) or {}
            data["names"] = {i: name for i, name in enumerate(cfg["classes"])}
            ds.yaml_path.write_text(
                yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            fixes.append({
                "image": "data.yaml",
                "line": "",
                "action": "rename_classes",
                "before": "names hiện tại",
                "after": str(cfg["classes"]),
            })

    return fixes
