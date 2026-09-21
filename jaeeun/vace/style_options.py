"""UI choices and model instructions shared by the upload workflow."""

STYLE_OPTIONS = {
    "bangs": {"풀뱅": "full blunt bangs", "시스루뱅": "wispy see-through bangs", "처피뱅": "short choppy baby bangs", "커튼뱅": "center-parted curtain bangs"},
    "length": {"단발": "short bob-length hair", "중간길이": "medium shoulder-length hair", "장발": "long hair below the shoulders"},
    "wave": {"생머리": "straight hair", "C컬": "C-shaped inward curled ends", "S컬(웨이브)": "S-shaped waves", "히피펌": "tight, voluminous hippie-perm curls"},
}
COLOR_OPTIONS = {
    "블랙": ("black", "블랙", "natural black"),
    "브라운": ("brown", "브라운", "natural brown"),
    "그레이": ("gray", "그레이", "silver gray"),
    "레드": ("red", "레드", "red"),
    "애쉬브라운": ("ash_brown", "애쉬 브라운", "cool ash brown"),
}


def inputs_ready(has_video: bool, options: dict, has_reference: bool) -> bool:
    return bool(has_video and (any(options.values()) or has_reference))


def hairstyle_prompt(options: dict, has_reference: bool) -> str:
    instructions = [f"Set the {key} to {values[options[key]]}."
                    for key, values in STYLE_OPTIONS.items() if options.get(key)]
    if has_reference:
        instructions.insert(0, "Use the reference image for the target hairstyle. Explicit selected attributes override the reference; use the reference for unspecified hairstyle attributes.")
    else:
        instructions.insert(0, "Edit the source person's hair using the selected attributes. Preserve unspecified hairstyle attributes from the source.")
    instructions.append("Preserve the source person's identity, face, expression, clothing and background.")
    return " ".join(instructions)


def candidate_palette(personal_color: str, requested_color: str | None, palettes: dict) -> tuple:
    recommended = palettes[personal_color]
    if not requested_color:
        return recommended
    selected = COLOR_OPTIONS[requested_color]
    return (selected,) + tuple(item for item in recommended if item[0] != selected[0])
