"""Frontend project helpers for ZFrontendBench."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from benchmarks.zfrontendbench.models import FrontendCheckResult


class ProjectType(StrEnum):
    HTML = "html"
    SVG = "svg"
    NPM = "npm"
    UNKNOWN = "unknown"


@dataclass
class ProjectInfo:
    project_type: ProjectType
    project_dir: Path
    framework: str = "unknown"


@dataclass
class BuildResult:
    success: bool
    project_type: ProjectType
    source_dir: Path | None = None
    entry_file: Path | None = None
    artifact_tar_path: Path | None = None
    server_side: bool = False
    error_message: str = ""
    code_failure: bool = False


FRAMEWORK_DEPS = {
    "next": "next",
    "nuxt": "nuxt",
    "gatsby": "gatsby",
    "astro": "astro",
    "@angular/core": "angular",
    "solid-js": "solid",
    "preact": "preact",
    "svelte": "svelte",
    "@sveltejs/kit": "svelte",
    "vue": "vue",
    "@vue/cli-service": "vue",
    "react": "react",
    "react-dom": "react",
    "react-scripts": "react",
}

DEFAULT_BUILD_DIRS = ["dist", "build", "out", "public", ".next", ".nuxt", ".output", "_site", "www"]


def detect_project(project_dir: Path) -> ProjectInfo:
    """Detect NPM/HTML/SVG project type and likely project root."""

    project_dir = Path(project_dir)
    npm = _find_npm_project(project_dir)
    if npm:
        npm_dir, pkg = npm
        return ProjectInfo(ProjectType.NPM, npm_dir, _detect_framework(pkg))

    svg_files = [p for p in project_dir.glob("**/*.svg") if "node_modules" not in p.parts]
    html_files = [p for p in project_dir.glob("**/*.html") if "node_modules" not in p.parts]
    if svg_files and not html_files:
        return ProjectInfo(ProjectType.SVG, project_dir)
    if html_files:
        return ProjectInfo(ProjectType.HTML, project_dir)
    return ProjectInfo(ProjectType.UNKNOWN, project_dir)


def find_project_root(workspace_path: Path, max_depth: int = 3) -> Path:
    current = Path(workspace_path)
    for _ in range(max_depth + 1):
        if (current / "package.json").exists() or (current / "index.html").exists():
            return current
        subdirs = [
            p
            for p in current.iterdir()
            if p.is_dir() and p.name not in {"node_modules", ".git", "__pycache__"}
        ]
        if len(subdirs) != 1:
            break
        current = subdirs[0]

    pkg_files = [p for p in workspace_path.glob("**/package.json") if "node_modules" not in p.parts]
    if pkg_files:
        return sorted(pkg_files, key=lambda p: len(p.parts))[0].parent
    html_files = [p for p in workspace_path.glob("**/index.html") if "node_modules" not in p.parts]
    if html_files:
        return sorted(html_files, key=lambda p: len(p.parts))[0].parent
    return Path(workspace_path)


def find_entry_html(project_dir: Path) -> Path | None:
    index_file = project_dir / "index.html"
    if index_file.exists():
        return index_file
    html_files = [p for p in project_dir.glob("*.html") if p.is_file()]
    if html_files:
        return html_files[0]
    html_files = [p for p in project_dir.glob("**/*.html") if "node_modules" not in p.parts]
    return html_files[0] if html_files else None


def get_unique_html_or_svg(project_dir: Path) -> Path | None:
    files = [
        p
        for p in project_dir.glob("**/*")
        if p.is_file() and p.suffix.lower() in {".html", ".svg"} and "node_modules" not in p.parts
    ]
    if not files:
        return None
    index_files = [p for p in files if p.name.lower() == "index.html"]
    return index_files[0] if index_files else files[0]


def wrap_svg_as_html(svg_file: Path) -> Path:
    html_file = svg_file.parent / "index.html"
    svg_content = svg_file.read_text(encoding="utf-8", errors="replace")
    html_file.write_text(
        "<!DOCTYPE html>\n<html><body>\n"
        f"{svg_content}\n"
        "</body></html>\n",
        encoding="utf-8",
    )
    return html_file


def parse_judge_score(output: str) -> tuple[float | None, str]:
    if not output:
        return None, "Empty judge output"
    if "该项目符合要求" in output and "该项目不符合要求" in output:
        return None, output
    if "判断结论：该项目符合要求" in output or "判断结论:该项目符合要求" in output:
        return 1.0, output
    if "判断结论：该项目不符合要求" in output or "判断结论:该项目不符合要求" in output:
        return 0.0, output
    if "该项目符合要求" in output and "请输出" not in output:
        return 1.0, output
    if "该项目不符合要求" in output and "请输出" not in output:
        return 0.0, output
    return None, output


def calculate_weighted_score(results: list[FrontendCheckResult]) -> float | None:
    if any(result.score is None for result in results):
        return None
    total_weight = sum(result.weight for result in results) or 1.0
    return sum((result.score or 0.0) * result.weight for result in results) / total_weight


def is_retriable_build_error(error_message: str) -> bool:
    lower = error_message.lower()
    if not lower:
        return True
    if "queue limit" in lower or "processconfig" in lower or "no build output found" in lower:
        return True
    if "workspace is empty" in lower:
        return True
    if "error during build" in lower or "cannot identify project type" in lower:
        return False
    return True


def _find_npm_project(project_dir: Path) -> tuple[Path, dict] | None:
    candidates = [project_dir / "package.json"]
    candidates.extend(
        p for p in project_dir.glob("**/package.json") if "node_modules" not in p.parts
    )
    for package_json in candidates:
        if not package_json.exists():
            continue
        try:
            pkg = json.loads(package_json.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "build" in pkg.get("scripts", {}) or "start" in pkg.get("scripts", {}):
            return package_json.parent, pkg
    return None


def _detect_framework(pkg: dict) -> str:
    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    for dep, framework in FRAMEWORK_DEPS.items():
        if dep in deps:
            return framework
    return "other"
