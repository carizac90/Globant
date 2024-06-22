from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, explode, regexp_replace, regexp_extract, trim, when, lower, round, lit, concat_ws, array_contains, coalesce, row_number, monotonically_increasing_id, udf
from pyspark.sql.types import StringType, FloatType, IntegerType, StructType, StructField
import pyspark.sql.functions as F
from pyspark.sql.window import Window 

# Iniciar Spark Session
spark = SparkSession.builder.appName("transformacion").getOrCreate()

# Diccionario de mapeo de marcas con sub-marcas
brand_mapping = {
    "victoria's secret": ("Victoria's Secret", ""),
    "victoria's secret pink": ("Victoria's Secret", "Pink"),
    "us topshop": ("Topshop", "Us Top Shop"),
    "calvin klein": ("Calvin Klein", ""),
    "hanky panky": ("Hanky Panky", ""),
    "b.tempt'd by wacoal": ("Wacoal", "b.tempt'd"),
    "wacoal": ("Wacoal", ""),
    "vanity fair": ("Vanity Fair", ""),
    "calvin klein modern cotton": ("Calvin Klein", "Modern Cotton"),
    "calvin klein performance": ("Calvin Klein", "Performance"),
    "b.tempt'd": ("Wacoal", "b.tempt'd"),
    "nordstrom lingerie": ("Nordstrom", ""),
    "hankypanky": ("Hanky Panky", ""),
    "calvin-klein": ("Calvin Klein", ""),
    "b-temptd": ("Wacoal", "b-temptd"),
    "hanky-panky": ("Hanky Panky", ""),
    "victorias-secret": ("Victoria's Secret", ""),
    "s": ("Wacoal", ""),
    "ref=w_bl_sl_l_b_ap_web_2603426011?ie=utf8&node=2603426011&field-lbr_brands_browse-bin=wacoal": ("Wacoal", ""),
    "ref=w_bl_sl_l_ap_ap_web_2586685011?ie=utf8&node=2586685011&field-lbr_brands_browse-bin=calvin+klein": ("Calvin Klein", ""),
    "ref=w_bl_sl_l_b_ap_web_2586451011?ie=utf8&node=2586451011&field-lbr_brands_browse-bin=b.tempt%27d": ("Wacoal", "b-temptd"),
    "lucky-brand": ("Lucky Brand", "Lucky-Brand"),
    "aerie": ("American Eagle", "Aerie"),
    "aeo": ("American Eagle", "Aeo")
}

# Función para mapear marcas
def map_brand(brand):
    brand = brand.lower()
    if brand in brand_mapping:
        mapped_brand = brand_mapping[brand]
        return {"brand_name": mapped_brand[0], "sub_brand_name": mapped_brand[1]}
    else:
        return {"brand_name": brand, "sub_brand_name": ""}
    
# Esquema para la estructura que devolverá la UDF
schema = StructType([
    StructField("brand_name", StringType(), True),
    StructField("sub_brand_name", StringType(), True)
])

# Convertir la función a UDF
map_brand_udf = udf(map_brand, schema)

# Función para cargar y preparar datos
def load_and_prepare_data(filepath):
    df = spark.read.json(filepath)
    df.printSchema()
    return df

# Función para contar valores nulos
def count_nulls(df):
    null_counts = df.select([(F.count(F.when(F.col(c).isNull(), c)).alias(c)) for c in df.columns]).toPandas()
    print(null_counts)

# Función para limpiar datos iniciales
def initial_data_clean(df):
    df_no_duplicates = df.dropDuplicates()
    print(f"Original row count: {df.count()}, Cleaned row count: {df_no_duplicates.count()}")
    return df_no_duplicates

# Función para asegurar que total_sizes y available_size sean arreglos
def ensure_array_columns(df):
    df = df.withColumn("total_sizes", split(col("total_sizes"), ",\\s*"))
    df = df.withColumn("available_size", split(col("available_size"), ",\\s*"))
    return df

# Función para normalizar valores de review_count (convirtiendo valores extremos a null)
def normalize_review_count(df):
    max_valid_value = 1e6  # Establece un valor máximo razonable para review_count
    df = df.withColumn("review_count", 
                       when(col("review_count") > max_valid_value, None)
                       .otherwise(col("review_count")).cast(FloatType()))
    return df

def normalize_and_explode_sizes(df):
    # Explode y limpieza básica de la columna size
    df = df.withColumn("size", explode(col("total_sizes")))
    df = df.withColumn("size", trim(regexp_replace(col("size"), r'\s+', '')))
    df = df.withColumn("size", regexp_replace(col("size"), r'\(.*?\)', ''))
    df = df.withColumn("size", regexp_replace(col("size"), r'[^a-zA-Z0-9]', ''))

    # Extracción de componentes numéricos y alfabéticos
    df = df.withColumn("size_number", regexp_extract(col("size"), r"(\d+)", 1).cast(IntegerType()))
    df = df.withColumn("size_letter", regexp_extract(col("size"), r"([a-zA-Z]+)$", 1))

    # Verificar si el tamaño está disponible
    df = df.withColumn("is_available", when(array_contains(col("available_size"), col("size")), 1).otherwise(0))

    # Condiciones de clasificación basadas únicamente en 'size'
    size_conditions = [
        # Grupos de tamaño basados en underbust size
        (col("size_number").between(30, 32), "Small"),
        (col("size_number").between(34, 36), "Medium"),
        (col("size_number").between(38, 40), "Large"),
        (col("size_number").between(42, 46), "Extra Large"),
        
        # No bras
        (col("size").rlike("(?i)^[SMLXLXXL]+$"), "Not Bras"),  # Ejemplo: S, M, L, XL, XXL, ...
        (col("size").rlike("(?i)^[6789]+$"), "Not Bras"),  # Ejemplo: 6, 7, 8, 9, ...
        (col("size").rlike("(?i)^[A-Z]+$"), "Not Bras")  # Ejemplo: A, B, C, D, ...
    ]

    df = df.withColumn(
        "size_group",
        coalesce(*[when(cond, group) for cond, group in size_conditions], lit("Unknown"))
    )
    
    return df

# Función para calcular métricas iniciales e incluir Warning_Sanction
def calculate_initial_metrics(df):
    df = df.withColumn("total_offered_items", F.size(col("total_sizes")).cast(IntegerType())) \
           .withColumn("available_items", F.size(col("available_size")).cast(IntegerType())) \
           .withColumn("availability_percentage", round((col("available_items") / col("total_offered_items")), 2).cast(FloatType())) \
           .withColumn("status", when(col("availability_percentage") < 0.3, "Sanction")
                                .when(col("availability_percentage") < 0.5, "Warning")
                                .otherwise("OK")) \
           .withColumn("Warning_Sanction",
                       when((col("size_group") == "Extra Large") & (col("availability_percentage") < 0.5) & (col("availability_percentage") >= 0.3), "Warning")
                       .when((col("size_group") == "Extra Large") & (col("availability_percentage") < 0.3), "Sanction")
                       .otherwise(lit("")))
    return df

# Listas de tallas a excluir si están en la categoría de "Bras" o "Bralette"
exclude_sizes = ["xs", "s", "m", "l", "xl", "xxl", "xlarge", "6", "7", "8", "9", "10", "2", "4", "12", 
                 "a", "b", "c", "d", "dd", "ddd", "xsmall", "14", "xxsmall", "aaa", "bc"]

# Función para asignar categorías
def assign_category(df):
    if "category" not in df.columns:
        df = df.withColumn("category", lit(None).cast("string"))

    category_patterns = [
        (r"(?i)\b(bra|push-up|plunge|demi|balconette|multi-way|Embellished Underwire Bra|bralette)\b", "Bras"),
        (r"(?i)\b(bandeau|triangle bralette|bralette)\b", "Tops"),
        (r"(?i)\b(hipster|cheeky|cheekster|boyshort|thong|hiphugger|boybrief|v-string|shortie|brief|panty|hipkini|cheekini|tanga|high cut brief|high cut briefs)\b", "Panties"),
        (r"(?i)\b(bikini)\b", "Bikinis"),
        (r"(?i)\b(vikini|v kini)\b", "Bikinis"),
        (r"(?i)\b(slip|chemise|babydoll|teddy|romper|camisole|bodysuit|bustier|garter belt)\b", "Sleepwear and Lingerie"),
        (r"(?i)\b(legging|short|jogger|tank|top|hoodie|tee|shirt|camisole|crop|sports bra|shorts|tank top)\b", "Tops Sport"),
        (r".*", "Other")
    ]

    for pattern, category in category_patterns:
        df = df.withColumn(
            "category",
            when(col("category").isNull() & col("product_name").rlike(pattern), category).otherwise(col("category"))
        )

    return df

# Función para asignar subcategorías adicionales
def assign_subcategories(df):
    subcategory_patterns = [
        (r"(?i)\b(seamless)\b", "Seamless"),
        (r"(?i)\b(lace)\b", "Lace"),
        (r"(?i)\b(mesh)\b", "Mesh"),
        (r"(?i)\b(crochet)\b", "Crochet"),
        (r"(?i)\b(embroidered|embroidery)\b", "Embroidered"),
        (r"(?i)\b(ruched)\b", "Ruched"),
        (r"(?i)\b(high-waist|high waist)\b", "High-Waist"),
        (r"(?i)\b(low-rise|low rise)\b", "Low-Rise"),
        (r"(?i)\b(mid-rise|mid rise)\b", "Mid-Rise"),
        (r"(?i)\b(adjustable)\b", "Adjustable"),
        (r"(?i)\b(push-up|push up)\b", "Push-Up"),
        (r"(?i)\b(padded)\b", "Padded"),
        (r"(?i)\b(unlined)\b", "Unlined"),
        (r"(?i)\b(wireless)\b", "Wireless"),
        (r"(?i)\b(nylon)\b", "Nylon"),
        (r"(?i)\b(spandex)\b", "Spandex"),
        (r"(?i)\b(cotton)\b", "Cotton"),
        (r"(?i)\b(satin)\b", "Satin"),
        (r"(?i)\b(polyamide)\b", "Polyamide"),
        (r"(?i)\b(elastane)\b", "Elastane")
    ]

    if "subcategory" not in df.columns:
        df = df.withColumn("subcategory", lit(None).cast("string"))

    for pattern, subcategory in subcategory_patterns:
        df = df.withColumn(
            "subcategory",
            when(
                col("subcategory").isNull() & 
                (col("product_name").rlike(pattern) | col("description").rlike(pattern)),
                subcategory
            ).otherwise(col("subcategory"))
        )
    return df

# Función para clasificar colores
def classify_color(color):
    colors = {
        "red": ["red", "ruby", "crimson", "candy apple"],
        "blue": ["blue", "navy", "cerulean", "indigo", "aqua", "cobalt"],
        "green": ["green", "teal", "jade", "mint", "basil", "olive"],
        "yellow": ["yellow", "gold", "lemon", "chartreuse"],
        "pink": ["pink", "fuchsia", "mauve", "rose", "bubblegum"],
        "purple": ["purple", "violet", "lavender", "plum", "amethyst"],
        "orange": ["orange", "peach", "coral", "apricot", "cinnamon"],
        "brown": ["brown", "taupe", "chocolate", "bronze", "cappuccino"],
        "black": ["black", "charcoal", "ebony"],
        "white": ["white", "ivory", "cream"],
        "grey": ["grey", "silver", "smokey", "slate"],
        "multi": ["multi"]
    }
    if color is None:
        return "unknown"
    color_lower = color.lower()
    for color_name, shades in colors.items():
        for shade in shades:
            if shade in color_lower:
                return color_name
    return "unknown"

# Convertir la función a UDF
classify_color_udf = udf(classify_color, StringType())

# Actualizar el proceso de datos para incluir el mapeo de marcas
def process_data():
    df = load_and_prepare_data("/home/carizac/carizac/challenge_Globant/Products.json")
    count_nulls(df)
    df_no_duplicates = initial_data_clean(df)
    
    # Asegurar que total_sizes y available_size sean arreglos
    df_no_duplicates = ensure_array_columns(df_no_duplicates)
    
    # Normalizar y explotar tamaños
    df_normalized = normalize_and_explode_sizes(df_no_duplicates)
    
    # Calcular métricas iniciales
    df_metrics = calculate_initial_metrics(df_normalized)
    
    # Normalizar review_count para valores extremos
    df_metrics = normalize_review_count(df_metrics)
    
    # Aplicar el mapeo de marcas
    df_metrics = df_metrics.withColumn("brand_name", lower(trim(col("brand_name"))))
    df_metrics = df_metrics.withColumn("brand_struct", map_brand_udf(col("brand_name")))
    df_metrics = df_metrics.withColumn("brand_name", col("brand_struct.brand_name"))
    df_metrics = df_metrics.withColumn("sub_brand_name", col("brand_struct.sub_brand_name"))
    df_metrics = df_metrics.drop("brand_struct")

    # Asignar categorías y subcategorías
    df_normalized = assign_category(df_metrics)
    df_normalized = assign_subcategories(df_normalized)

    # Filtrar los productos en la categoría "Bras" o "Bralette" con las tallas a excluir
    df_filtered = df_normalized.filter(
        ~((col("category") == "Bras") & (col("size").isin(exclude_sizes))) &
        ~((col("category") == "Bralette") & (col("size").isin(exclude_sizes)))
    )

    # Clasificar colores
    df_filtered = df_filtered.withColumn("color_group", classify_color_udf(col("color")))

    # Crear las dimensiones y hechos del esquema Snowflake
    windowSpec = Window.orderBy(monotonically_increasing_id())
    df_filtered = df_filtered.withColumn("product_id", row_number().over(windowSpec).cast(IntegerType()))
    df_filtered = df_filtered.withColumn("retailer_id", row_number().over(windowSpec).cast(IntegerType()))
    df_filtered = df_filtered.withColumn("size_id", row_number().over(windowSpec).cast(IntegerType()))

    dim_product = df_filtered.select("brand_name", "sub_brand_name", "color", "product_name", "category", "subcategory", "product_id","is_available").distinct()

    dim_retailer = df_filtered.select("retailer", "retailer_id").distinct()

    dim_size = df_filtered.select("size", "size_number", "size_letter", "is_available", "size_id", "size_group").distinct()

    fact_product_sales = df_filtered \
        .select(
            row_number().over(windowSpec).alias("sales_id").cast(IntegerType()),
            col("product_id"),
            col("retailer_id"),
            col("size_id"),
            col("rating").cast(FloatType()),
            col("review_count").cast(FloatType()),
            col("is_available"),
            col("mrp").cast(FloatType()).alias("sales_amount"),
            col("total_offered_items").cast(IntegerType()),
            col("available_items").cast(IntegerType()),
            col("availability_percentage"),
            col("status"),
            col("Warning_Sanction")
        ).distinct()

    dim_product.printSchema()
    dim_retailer.printSchema()
    dim_size.printSchema()
    fact_product_sales.printSchema()

    dim_product.show(truncate=False)
    dim_retailer.show(truncate=False)
    dim_size.show(truncate=False)
    fact_product_sales.show(truncate=False)

    df_filtered = df_filtered.withColumn("available_size", concat_ws(",", col("available_size")))
    df_filtered = df_filtered.withColumn("total_sizes", concat_ws(",", col("total_sizes")))

    df_filtered.printSchema()

    output_paths = {
        "dim_product": '/home/carizac/dim_product.csv',
        "dim_retailer": '/home/carizac/dim_retailer.csv',
        "dim_size": '/home/carizac/dim_size.csv',
        "fact_product_sales": '/home/carizac/fact_product_sales.csv'
    }

    dim_product.write.csv(output_paths["dim_product"], header=True, mode="overwrite")
    dim_retailer.write.csv(output_paths["dim_retailer"], header=True, mode="overwrite")
    dim_size.write.csv(output_paths["dim_size"], header=True, mode="overwrite")
    fact_product_sales.write.csv(output_paths["fact_product_sales"], header=True, mode="overwrite")

    for table, path in output_paths.items():
        print(f"Archivo CSV de {table} exportado a: {path}")
        
    # Identificar tallas fuera de los grupos predefinidos
    predefined_groups = ["Small", "Medium", "Large", "Extra Large", "Unknown", 
                         "XXS", "XS", "S", "M", "L", "XL", "XXL", "true", 
                         "XLarge", "2426Plus", "1XPlus", "1XApparel", 
                         "2XApparel", "3XApparel", "false", "OK"]
    
    unique_sizes = df_filtered.select("size_group").distinct()
    unusual_sizes = unique_sizes.filter(~col("size_group").isin(predefined_groups))
    
    unusual_sizes.show(truncate=False)

# Ejecutar el procesamiento de datos
process_data()

# Cerrar la sesión Spark al finalizar
spark.stop()
