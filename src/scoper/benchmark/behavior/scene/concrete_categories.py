CONCRETE_CATEGORY_TYPES = {'license_plate': 'LicensePlate', 'spotlight': 'Spotlight', 'plant_pot': 'PlantPot', 'pewter_teapot': 'Teapot', 'bookend': 'Bookend', 'knife_block': 'KnifeBlock', 'pencil_case': 'PencilCase', 'pencil_holder': 'PencilHolder', 'cup_holder': 'CupHolder', 'potato_masher': 'PotatoMasher', 'bar_soap': 'SoapBar', 'apple_pie': 'ApplePie', 'bottle_of_apple_cider': 'AppleCiderBottle', 'bottle_of_apple_juice': 'AppleJuiceBottle', 'bottle_of_tomato_paste': 'TomatoPasteBottle', 'bottle_of_tomato_juice': 'TomatoJuiceBottle', 'tomato_paste_can': 'TomatoPasteCan', 'cup_of_yogurt': 'YogurtCup', 'easter_egg': 'EasterEgg', 'cooking_oil_atomizer': 'CookingOilSprayer', 'vinegar_atomizer': 'VinegarSprayer', 'pesticide_atomizer': 'PesticideSprayer'}
PORTABLE_TYPES = set(CONCRETE_CATEGORY_TYPES.values()) - {'Spotlight'}
CONCRETE_GROUP_ADDITIONS = {'pickupable_task_objects': PORTABLE_TYPES, 'cleaning_supplies': {'SoapBar'}, 'dishware': {'Teapot'}, 'dirtyable_task_objects': {'Teapot'}, 'cold_food': {'ApplePie', 'AppleJuiceBottle', 'TomatoJuiceBottle', 'TomatoPasteBottle', 'TomatoPasteCan', 'YogurtCup'}, 'coolable_task_objects': {'ApplePie', 'AppleJuiceBottle', 'TomatoJuiceBottle', 'TomatoPasteBottle', 'TomatoPasteCan', 'YogurtCup'}, 'heatable_task_objects': {'ApplePie'}}

def apply_concrete_public_names(task):
    for name, role in task.get('roles', {}).items():
        category = role.get('category')
        if category not in CONCRETE_CATEGORY_TYPES:
            continue
        role['object_type'] = CONCRETE_CATEGORY_TYPES[category]
        override = task.setdefault('binding_policy', {}).setdefault('roles', {}).setdefault(name, {})
        override['public_terms'] = [category.replace('_', ' ')]
    return task
