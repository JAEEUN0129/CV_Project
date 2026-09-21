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

BANGS_INSTRUCTIONS = {
    "풀뱅": "Create a dense, continuous blunt fringe across the forehead, ending just above the eyebrows. No centre part or open curtain gap.",
    "시스루뱅": "Create sparse, fine separated wispy bangs across the forehead with visible skin between strands. Do not create a thick solid fringe.",
    "처피뱅": "Create short choppy baby bangs ending high on the forehead, clearly above the eyebrows, with an irregular textured edge. Leave a visible strip of bare forehead below the fringe. No eyebrow-length full bangs.",
    "커튼뱅": "Remove the full horizontal fringe. Create a clearly open centre part with visible central forehead. Sweep two symmetric fringe sections outward toward the temples, gradually lengthening at the sides. No straight-across blunt bangs and no solid fringe covering the centre forehead.",
}


def regional_prompt(options: dict, region: str) -> str:
    preserve = " Preserve this image's exact identity, facial features, hair colour, lighting and background."
    if region == "bangs":
        if options["bangs"] == "커튼뱅":
            return (BANGS_INSTRUCTIONS["커튼뱅"]
                    + " Edit the complete front fringe and temple sections together. The exposed centre forehead must connect continuously from the centre part down to the eyebrows, without an isolated skin hole or a remaining horizontal strip of old bangs. Blend the swept sections into the existing side hair; preserve the lower haircut."
                    + preserve)
        return BANGS_INSTRUCTIONS[options["bangs"]] + " Edit only the forehead fringe; preserve all side and lower hair." + preserve
    body_options = {key: value for key, value in options.items() if key in {"length", "wave"}}
    return (hairstyle_prompt(body_options, False)
            + " Edit only the side and lower hair. Remove old strands outside the requested haircut and reconstruct the revealed skin, clothing or background; do not leave a second layer of old hair below the new ends. The existing forehead fringe and top centre hair are locked and must not change." + preserve)


def inputs_ready(has_video: bool, options: dict, has_reference: bool) -> bool:
    return bool(has_video and (any(options.values()) or has_reference))


def hairstyle_prompt(options: dict, has_reference: bool) -> str:
    instructions = [f"Set the {key} to {values[options[key]]}."
                    for key, values in STYLE_OPTIONS.items() if options.get(key)]
    if has_reference:
        specified = ", ".join(key for key in STYLE_OPTIONS if options.get(key))
        instructions.insert(0, "The text-selected attributes are mandatory. Use the reference ONLY for unspecified hairstyle attributes. "
                            + (f"Do not copy the reference's {specified}; replace those attributes with the following instructions." if specified else "Match the reference hairstyle."))
        for field in STYLE_OPTIONS:
            if not options.get(field):
                instructions.append(f"Copy the {field} from the reference image, not from the source person.")
        if not options.get("bangs"):
            instructions.append("Match the reference fringe, including its presence or absence, coverage, density and parting. If the reference has bangs, create those bangs over the forehead; do not retain an exposed source forehead.")
    else:
        instructions.insert(0, "Edit the source person's hair using the selected attributes. Preserve unspecified hairstyle attributes from the source.")
    if options.get("length") == "장발":
        instructions.append("The hair must extend visibly below the shoulders toward the chest. Do not produce a bob or shoulder-length haircut.")
    if options.get("wave") == "C컬":
        instructions.append("Keep the upper lengths smooth and mostly straight, with a single inward C-shaped bend at the ends. Do not create S-waves, repeated waves or tight curls.")
    if options.get("bangs"):
        instructions.append(BANGS_INSTRUCTIONS[options["bangs"]])
    instructions.append("Preserve the source person's identity, face, expression, clothing and background.")
    return " ".join(instructions)


def requested_style_label(options: dict, has_reference: bool = False) -> str:
    selected = [options[key] for key in ("bangs", "length", "wave", "color") if options.get(key)]
    selected.append("스타일사진 O" if has_reference else "스타일사진 X")
    return f"요청한 스타일 ({' · '.join(selected)})"
