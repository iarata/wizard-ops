Dish Metadata
The dish metadata CSVs (metadata/dish_metadata_cafe1.csv and metadata/dish_metadata_cafe2.csv) contain all nutrition metadata at the dish-level, as well as per-ingredient mass and macronutrients. For each dish ID dish_[10 digit timestamp], there is a CSV entry containing the following fields:

dish_id, total_calories, total_mass, total_fat, total_carb, total_protein, num_ingrs, (ingr_1_id, ingr_1_name, ingr_1_grams, ingr_1_calories, ingr_1_fat, ingr_1_carb, ingr_1_protein, ...)

with the last 8 fields are repeated for every ingredient present in the dish.