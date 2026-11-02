"""Personalized FitMaxxx coaching using aggregate session data only."""
from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """You are FitMaxxx, a practical and encouraging fitness companion.
Use only the supplied profile and aggregate activity summary. Never invent readings,
claims, targets, diagnoses, or progress. Give 3 concise sections: What stands out,
What to focus on next, and A small action for today. Adapt tone and suggestions to
the user's stated goal and fitness level. Keep advice general and non-medical. Do not
recommend extreme calorie restriction, unsafe training, or ignoring pain. If sleep or
activity information is missing, say it is not available rather than guessing. Maximum
120 words. Return clean Markdown without generic disclaimers or implementation notes."""


def build_context(profile: dict[str, Any], wearable: dict[str, Any] | None, activity: dict[str, Any] | None = None) -> dict[str, Any]:
    profile_fields = {
        "display_name": profile.get("display_name", "there"),
        "age": profile.get("age"),
        "weight_kg": profile.get("weight_kg"),
        "goal": profile.get("goal"),
        "fitness_level": profile.get("fitness_level"),
        "steps_goal": profile.get("steps_goal"),
        "active_goal_min": profile.get("active_goal_min"),
    }
    wearable_fields = None
    if wearable:
        wearable_fields = {
            key: wearable.get(key) for key in (
                "device", "steps", "steps_goal", "active_minutes", "active_goal_min",
                "sleep_hours", "current_hr", "resting_hr", "calories_kcal", "activity_score"
            )
        }
    activity_fields = None
    if activity:
        activity_fields = {
            "duration_min": round(float(activity.get("duration_min", 0)), 1),
            "walking_min": round(float(activity.get("active_min", 0)), 1),
            "estimated_kcal": round(float(activity.get("kcal", 0)), 1),
            "top_activity": activity.get("top_activity"),
            "mean_confidence_percent": round(float(activity.get("confidence", 0)) * 100),
        }
    return {"profile": profile_fields, "wearable_summary": wearable_fields, "activity_summary": activity_fields}


def local_coach_note(context: dict[str, Any]) -> str:
    """Deterministic fallback that is personalized without calling an external service."""
    p = context["profile"]
    w = context.get("wearable_summary") or {}
    name = p.get("display_name") or "there"
    goal = p.get("goal") or "Build consistency"
    level = p.get("fitness_level") or "Beginner"
    lines = [
        f"**A good check-in, {name}.**",
        f"Your current focus is **{goal.lower()}**.",
    ]

    if w:
        steps = int(w.get("steps") or 0)
        goal_steps = max(1, int(w.get("steps_goal") or p.get("steps_goal") or 8000))
        active = int(w.get("active_minutes") or 0)
        active_goal = max(1, int(w.get("active_goal_min") or p.get("active_goal_min") or 45))
        sleep = w.get("sleep_hours")
        lines.append(f"You've logged **{steps:,} steps** ({min(steps / goal_steps * 100, 999):.0f}% of your daily target) and **{active} active minutes**.")
        if sleep is not None:
            lines.append(f"Last night's recorded sleep was **{float(sleep):.1f} hours**. Keep your sleep and activity routines consistent.")
        if active < active_goal:
            lines.append(f"For your **{goal.lower()}** goal, a relaxed **10–15 minute walk** is a manageable way to add movement today.")
        else:
            lines.append(f"You've reached your active-minute target. Keep the rest of the day balanced with recovery and regular movement.")
    else:
        lines.append(f"Your current focus is **{goal.lower()}** at a **{level.lower()}** level.")
        lines.append("Add one small, repeatable movement block to your day and build from a routine you can sustain.")
    return "\n\n".join(lines)


def claude_note(context: dict[str, Any], api_key: str, model: str = "claude-haiku-5-5", timeout: int = 25) -> str | None:
    """Ask Claude for a note. Only small aggregate numbers/profile fields are sent."""
    if not api_key:
        return None
    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        result = client.messages.create(
            model=model,
            max_tokens=450,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(context, ensure_ascii=False)}],
        )
        text = "\n".join(
            block.text for block in result.content if getattr(block, "type", "") == "text"
        ).strip()
        return text or None
    except Exception:
        return None


def generate_note(context: dict[str, Any], api_key: str = "", model: str = "claude-haiku-5-5") -> tuple[str, str]:
    """Return (note, source_label); silently fall back when Claude is not configured."""
    generated = claude_note(context, api_key, model=model) if api_key else None
    if generated:
        return generated, "AI-assisted"
    return local_coach_note(context), "Personalized insight"


def answer_question(question: str, context: dict[str, Any], api_key: str = "", model: str = "claude-haiku-5-5") -> str:
    """Answer a short coaching question using Claude when configured, otherwise grounded local guidance."""
    question = question.strip()
    if not question:
        return "Tell me what you would like to work on today."
    if api_key:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key, timeout=25)
            result = client.messages.create(
                model=model,
                max_tokens=400,
                system=SYSTEM_PROMPT + " Answer the user's one question directly, using the provided context.",
                messages=[{"role": "user", "content": json.dumps({"context": context, "question": question}, ensure_ascii=False)}],
            )
            text = "\n".join(block.text for block in result.content if getattr(block, "type", "") == "text").strip()
            if text:
                return text
        except Exception:
            pass
    profile = context.get("profile", {})
    wearable = context.get("wearable_summary") or {}
    goal = str(profile.get("goal") or "Build consistency").lower()
    steps = int(wearable.get("steps") or 0)
    target = max(1, int(wearable.get("steps_goal") or profile.get("steps_goal") or 8000))
    q = question.lower()
    if any(word in q for word in ("step", "walk", "walking")):
        remaining = max(target - steps, 0)
        if remaining == 0:
            return f"You've reached your {target:,}-step target in this demo snapshot. Keep your next movement break comfortable and consistent with your {goal} goal."
        return f"You are about {remaining:,} steps from your demo target. Try splitting that into two relaxed 10-minute walks rather than doing it all at once. That supports your goal to {goal}."
    if any(word in q for word in ("sleep", "rest", "bed")):
        sleep = wearable.get("sleep_hours")
        if sleep is None:
            return "Sleep data is not available in this session. A consistent wind-down and sleep schedule is a useful habit to track over time."
        return f"The simulated device recorded {float(sleep):.1f} hours of sleep. Look at the trend across several nights rather than judging one reading; consistency is a good starting point for your {goal} goal."
    if any(word in q for word in ("workout", "exercise", "routine", "plan")):
        return f"For your {goal} goal and {str(profile.get('fitness_level') or 'current').lower()} fitness level, start with a repeatable session: 5 minutes easy warm-up, 15–20 minutes at a comfortable effort, then a short cool-down. Adjust to how you feel."
    return f"Based on your {goal} goal, pick one small action you can repeat this week. Your current demo snapshot shows {steps:,} steps; use a few short movement breaks to build consistency, then review your progress."

# Backward-compatible helpers retained for the V2 project tests and reports.
def build_stats(metrics, summary_minutes, confidence, longest, nudges, impacts, placement, source):
    return {
        "source": source,
        "placement": placement,
        "duration_minutes": round(metrics["duration_min"], 1),
        "walking_minutes": round(metrics["active_min"], 1),
        "on_feet_minutes": round(metrics["upright_min"], 1),
        "sitting_or_laying_minutes": round(metrics["sedentary_min"], 1),
        "estimated_kcal": round(metrics["kcal"], 1),
        "movement_score": metrics["score"],
        "minutes_per_activity": {k: round(float(v), 2) for k, v in summary_minutes.items()},
        "model_confidence_percent": round(100 * confidence),
        "longest_still_stretch_minutes": round(longest["minutes"], 1) if longest else 0,
        "still_stretch_alerts": len(nudges),
        "impact_events": len(impacts),
    }


def rule_based_note(stats):
    top = max(stats["minutes_per_activity"], key=stats["minutes_per_activity"].get)
    total = max(sum(stats["minutes_per_activity"].values()), 1e-9)
    share = 100 * stats["minutes_per_activity"][top] / total
    lines = [
        f"- Most of this session was **{top.replace('_', ' ').lower()}** ({share:.0f}%), "
        f"with {stats['walking_minutes']} min of walking and a movement score of {stats['movement_score']}/100."
    ]
    if stats["longest_still_stretch_minutes"] >= 1:
        lines.append(
            f"- Your longest unbroken sitting or laying stretch was {stats['longest_still_stretch_minutes']} min"
            + (f", with {stats['still_stretch_alerts']} stretch(es) over your alert limit." if stats["still_stretch_alerts"] else ".")
        )
    else:
        lines.append("- You did not stay still for long at any point, so there was no long sitting stretch.")
    own_off_waist = stats["placement"] != "waist" and stats["source"] != "demo"
    if stats["model_confidence_percent"] < 70 or own_off_waist:
        reasons = [f"model confidence was {stats['model_confidence_percent']}%"]
        if own_off_waist:
            reasons.append("a phone not worn at the waist can behave differently from the training data")
        lines.append("- Treat this with care: " + ", and ".join(reasons) + ", so some labels may be wrong.")
    elif stats["walking_minutes"] < 0.3 * stats["duration_minutes"]:
        lines.append("- A short walk or a few stairs between sitting blocks would lift your score the most.")
    else:
        lines.append("- Good mix of movement. Keep breaking up any long sitting blocks with a minute on your feet.")
    return "\n".join(lines)


def llm_note(stats, api_key, model="claude-haiku-5-5", timeout=25):
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
        reply = client.messages.create(
            model=model,
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": json.dumps(stats)}],
        )
        text = "\n".join(block.text for block in reply.content if getattr(block, "type", "") == "text").strip()
        return text or None
    except Exception:
        return None
