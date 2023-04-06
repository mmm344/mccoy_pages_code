{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "ddacdb4d",
   "metadata": {},
   "source": [
    "# Phase I\n",
    "In this phase we import the raw data and perform essential data exploration and preprocessing"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b37348f3",
   "metadata": {},
   "source": [
    "Fix the cell size to maximize visable code per line"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3c4e6125",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>\n",
       "    .container { width:100% !important; }\n",
       "</style>\n"
      ],
      "text/plain": [
       "<IPython.core.display.HTML object>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "%%html\n",
    "<style>\n",
    "    .container { width:100% !important; }\n",
    "</style>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6d2c7b5f",
   "metadata": {},
   "source": [
    "Let's import the libraries we will need for this initial analysis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "d1853916",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]\n",
      "C:\\Users\\mmccoy\\Anaconda3\\python.exe\n",
      "C:\\Users\\mmccoy\\AppData\\Local\\Microsoft\\WindowsApps\\python.exe\n"
     ]
    }
   ],
   "source": [
    "%run relevant_libraries_phase_1.ipynb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "671a3185",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "os.environ['PYSPARK_PYTHON'] = 'C:/Users/mmccoy/AppData/Local/Microsoft/WindowsApps/python.exe'\n",
    "os.environ['PYSPARK_DRIVER_PYTHON'] = 'C:/Users/mmccoy/AppData/Local/Microsoft/WindowsApps/python.exe'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "14556a15",
   "metadata": {},
   "outputs": [],
   "source": [
    "spark = SparkSession.builder.appName(\"World Happiness Report\").getOrCreate()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0f5026ef",
   "metadata": {},
   "source": [
    "## Preprocessing/Cleaning\n",
    "____________________________________________________________________________________________"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "06261f87",
   "metadata": {},
   "source": [
    "Import the raw data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "2a949d91",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = spark.read.csv(\"../2022.csv\", header = True, inferSchema = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1267170d",
   "metadata": {},
   "source": [
    "____________________________________________________________________________________________\n",
    "Check the data types to see if any adjustments need to be made."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "a8c7a82f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('RANK', 'int'),\n",
       " ('Country', 'string'),\n",
       " ('Happiness score', 'string'),\n",
       " ('Whisker-high', 'string'),\n",
       " ('Whisker-low', 'string'),\n",
       " ('Dystopia (1.83) + residual', 'string'),\n",
       " ('Explained by: GDP per capita', 'string'),\n",
       " ('Explained by: Social support', 'string'),\n",
       " ('Explained by: Healthy life expectancy', 'string'),\n",
       " ('Explained by: Freedom to make life choices', 'string'),\n",
       " ('Explained by: Generosity', 'string'),\n",
       " ('Explained by: Perceptions of corruption', 'string')]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Let's take a look at the columns and their types\n",
    "df.dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "892ce533",
   "metadata": {},
   "source": [
    "Notice that we need to convert some string types to float types for machine learning purposes. Let's first rename our columns so they're easier to work with and we don't have to worry about special characters."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "b69f4360",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = (df.withColumnRenamed(\"RANK\", \"rank\")\n",
    "      .withColumnRenamed(\"Country\", \"country\")\n",
    "      .withColumnRenamed(\"Happiness score\", \"happiness_score\")\n",
    "      .withColumnRenamed(\"Whisker-high\", \"whisker_high\")\n",
    "      .withColumnRenamed(\"Whisker-low\", \"whisker_low\")\n",
    "      .withColumnRenamed(\"Dystopia (1.83) + residual\", \"dystopia_183_residual\")\n",
    "      .withColumnRenamed(\"Explained by: GDP per capita\", \"gdp_per_capita\")\n",
    "      .withColumnRenamed(\"Explained by: Social support\", \"social_support\")\n",
    "      .withColumnRenamed(\"Explained by: Healthy life expectancy\", \"healthy_life_expectancy\")\n",
    "      .withColumnRenamed(\"Explained by: Freedom to make life choices\", \"freedom_to_make_life_choices\")\n",
    "      .withColumnRenamed(\"Explained by: Generosity\", \"generosity\")\n",
    "      .withColumnRenamed(\"Explained by: Perceptions of corruption\", \"perception_of_corruption\")\n",
    "      .withColumn('country', when(col('country') == 'Taiwan Province of China', 'Taiwan').otherwise(col('country')))\n",
    "      .withColumn('country', when(col('country') == 'Hong Kong S.A.R. of China', 'Hong Kong').otherwise(col('country')))\n",
    "     )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "b2506bdc",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('rank', 'int'),\n",
       " ('country', 'string'),\n",
       " ('happiness_score', 'string'),\n",
       " ('whisker_high', 'string'),\n",
       " ('whisker_low', 'string'),\n",
       " ('dystopia_183_residual', 'string'),\n",
       " ('gdp_per_capita', 'string'),\n",
       " ('social_support', 'string'),\n",
       " ('healthy_life_expectancy', 'string'),\n",
       " ('freedom_to_make_life_choices', 'string'),\n",
       " ('generosity', 'string'),\n",
       " ('perception_of_corruption', 'string')]"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "5050629c",
   "metadata": {},
   "outputs": [],
   "source": [
    "#It looks like we need to remove commas to get our data types correct\n",
    "numeric_type = DoubleType()\n",
    "for col_name in df.columns:\n",
    "    # Replace the \",\" with \".\"\n",
    "    df = df.withColumn(col_name, regexp_replace(col_name, \",\", \".\"))\n",
    "    \n",
    "for col_name in df.columns[2:]:\n",
    "    df = df.withColumn(col_name, col(col_name).cast(numeric_type))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "b2b54f3c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "[('rank', 'string'),\n",
       " ('country', 'string'),\n",
       " ('happiness_score', 'double'),\n",
       " ('whisker_high', 'double'),\n",
       " ('whisker_low', 'double'),\n",
       " ('dystopia_183_residual', 'double'),\n",
       " ('gdp_per_capita', 'double'),\n",
       " ('social_support', 'double'),\n",
       " ('healthy_life_expectancy', 'double'),\n",
       " ('freedom_to_make_life_choices', 'double'),\n",
       " ('generosity', 'double'),\n",
       " ('perception_of_corruption', 'double')]"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.dtypes"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b3151912",
   "metadata": {},
   "source": [
    "____________________________________________________________________________________________\n",
    "Investigate NA values relative to ``happiness_score``"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "6c9b9ee7",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "+----+-------+---------------+------------+-----------+---------------------+--------------+--------------+-----------------------+----------------------------+----------+------------------------+\n",
      "|rank|country|happiness_score|whisker_high|whisker_low|dystopia_183_residual|gdp_per_capita|social_support|healthy_life_expectancy|freedom_to_make_life_choices|generosity|perception_of_corruption|\n",
      "+----+-------+---------------+------------+-----------+---------------------+--------------+--------------+-----------------------+----------------------------+----------+------------------------+\n",
      "|null|   null|           null|        null|       null|                 null|          null|          null|                   null|                        null|      null|                    null|\n",
      "+----+-------+---------------+------------+-----------+---------------------+--------------+--------------+-----------------------+----------------------------+----------+------------------------+\n",
      "\n",
      "+-------+-----+\n",
      "|country|count|\n",
      "+-------+-----+\n",
      "|     xx|    1|\n",
      "+-------+-----+\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# Count the number of NA values in each column\n",
    "na_counts = df.select([sum(when(isnan(c), 1)).alias(c) for c in df.columns])\n",
    "\n",
    "# Display the NA counts for each column\n",
    "na_counts.show()\n",
    "\n",
    "# Filter the rows where happiness_score is null and group by country\n",
    "null_scores_by_country = df.filter(df.happiness_score.isNull()) \\\n",
    "                          .groupBy(\"country\") \\\n",
    "                          .count()\n",
    "\n",
    "# Display the countries with null happiness scores and the number of occurrences\n",
    "null_scores_by_country.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3bca9d5e",
   "metadata": {},
   "source": [
    "Drop NA values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "ac48eacf",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Drop all rows with any null values\n",
    "df = df.na.drop()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b5141267",
   "metadata": {},
   "source": [
    "____________________________________________________________________________________________\n",
    "Let's see if we have any duplicate rows"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "04b9c4f5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "+----+-------+---------------+------------+-----------+---------------------+--------------+--------------+-----------------------+----------------------------+----------+------------------------+-----+\n",
      "|rank|country|happiness_score|whisker_high|whisker_low|dystopia_183_residual|gdp_per_capita|social_support|healthy_life_expectancy|freedom_to_make_life_choices|generosity|perception_of_corruption|count|\n",
      "+----+-------+---------------+------------+-----------+---------------------+--------------+--------------+-----------------------+----------------------------+----------+------------------------+-----+\n",
      "+----+-------+---------------+------------+-----------+---------------------+--------------+--------------+-----------------------+----------------------------+----------+------------------------+-----+\n",
      "\n"
     ]
    }
   ],
   "source": [
    "duplicate_row = df.groupBy(df.columns).count().where(col(\"count\") > 1).show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "23470308",
   "metadata": {},
   "source": [
    "We can drop the duplicates, but we don't see any for this data set. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "68945e6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.dropDuplicates()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d4d35cf7",
   "metadata": {},
   "source": [
    "____________________________________________________________________________________________\n",
    "Let's encode the categorical variables. We will use one-hot encoding."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "6b9397ed",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Index the string column\n",
    "indexer = StringIndexer(inputCol=\"country\", outputCol=\"country_index\")\n",
    "indexed = indexer.fit(df).transform(df)\n",
    "\n",
    "# One-hot encode the indexed column\n",
    "encoder = OneHotEncoder(inputCols=[\"country_index\"],\n",
    "                                 outputCols=[\"country_onehot\"])\n",
    "encoded = encoder.fit(indexed).transform(indexed)\n",
    "\n",
    "# Drop the original categorical column and the index column\n",
    "encoded = encoded.drop(\"country\", \"country_index\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bd1a6c52",
   "metadata": {},
   "source": [
    "Next, we will standardize df. Recall that df is a Spark ``pyspark.sql.dataframe.DataFrame``. We will do this in the following steps:\n",
    "1. Select the columns we want to normalize \n",
    "2. Assemble the selected columns into a vector column\n",
    "3. Scale and normalize the vector column\n",
    "4. Drop the original columns and keep only the normalized features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "fe3aab93",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "root\n",
      " |-- rank: string (nullable = true)\n",
      " |-- country: string (nullable = true)\n",
      " |-- happiness_score: double (nullable = true)\n",
      " |-- whisker_high: double (nullable = true)\n",
      " |-- whisker_low: double (nullable = true)\n",
      " |-- dystopia_183_residual: double (nullable = true)\n",
      " |-- gdp_per_capita: double (nullable = true)\n",
      " |-- social_support: double (nullable = true)\n",
      " |-- healthy_life_expectancy: double (nullable = true)\n",
      " |-- freedom_to_make_life_choices: double (nullable = true)\n",
      " |-- generosity: double (nullable = true)\n",
      " |-- perception_of_corruption: double (nullable = true)\n",
      "\n"
     ]
    }
   ],
   "source": [
    "df.printSchema()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "23df5a7b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pyspark.ml.feature import VectorAssembler, StandardScaler\n",
    "from pyspark.sql.functions import udf, struct, col\n",
    "from pyspark.sql.types import StructType, StructField, DoubleType\n",
    "\n",
    "#select the columns to normalize\n",
    "num_cols = df.columns[2:]\n",
    "\n",
    "# vectorize the features\n",
    "assembler = VectorAssembler(inputCols=num_cols, outputCol=\"features\")\n",
    "df_vector = assembler.transform(df)\n",
    "\n",
    "# standardize the features\n",
    "scaler = StandardScaler(inputCol=\"features\", outputCol=\"scaledFeatures\")\n",
    "scaler_model = scaler.fit(df_vector)\n",
    "df_scaled = scaler_model.transform(df_vector)\n",
    "\n",
    "# convert Vector column to StructType column\n",
    "vector_to_struct = udf(lambda v: struct([float(x) for x in v]), StructType([StructField(col, DoubleType()) for col in num_cols]))\n",
    "df_scaled = df_scaled.withColumn(\"scaledFeatures\", vector_to_struct(col(\"scaledFeatures\")))\n",
    "\n",
    "# # select the necessary columns\n",
    "# selected_cols = num_cols\n",
    "# for i in range(len(selected_cols)):\n",
    "#     df_scaled = df_scaled.withColumn(selected_cols[i], col(selected_cols[i]))\n",
    "\n",
    "# # drop the original columns and the scaled features column\n",
    "# df_scaled = df_scaled.drop(*num_cols).drop(\"features\").drop(\"scaledFeatures\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "c0065985",
   "metadata": {},
   "outputs": [
    {
     "ename": "PythonException",
     "evalue": "\n  An exception was thrown from the Python worker. Please see the stack trace below.\nTraceback (most recent call last):\n  File \"C:\\Users\\mmccoy\\Anaconda3\\Lib\\site-packages\\pyspark\\python\\lib\\pyspark.zip\\pyspark\\worker.py\", line 540, in main\nRuntimeError: Python in worker has different version 3.10 than that in driver 3.9, PySpark cannot run with different minor versions. Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set.\n",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mPythonException\u001b[0m                           Traceback (most recent call last)",
      "\u001b[1;32m~\\AppData\\Local\\Temp/ipykernel_18060/1861838437.py\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m \u001b[0mnew_df\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mdf_scaled\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mtoPandas\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pyspark\\sql\\pandas\\conversion.py\u001b[0m in \u001b[0;36mtoPandas\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    203\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    204\u001b[0m         \u001b[1;31m# Below is toPandas without Arrow optimization.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 205\u001b[1;33m         \u001b[0mpdf\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mpd\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mDataFrame\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfrom_records\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcollect\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mcolumns\u001b[0m\u001b[1;33m=\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    206\u001b[0m         \u001b[0mcolumn_counter\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mCounter\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcolumns\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    207\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pyspark\\sql\\dataframe.py\u001b[0m in \u001b[0;36mcollect\u001b[1;34m(self)\u001b[0m\n\u001b[0;32m    815\u001b[0m         \"\"\"\n\u001b[0;32m    816\u001b[0m         \u001b[1;32mwith\u001b[0m \u001b[0mSCCallSiteSync\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_sc\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 817\u001b[1;33m             \u001b[0msock_info\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m_jdf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcollectToPython\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    818\u001b[0m         \u001b[1;32mreturn\u001b[0m \u001b[0mlist\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0m_load_from_socket\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0msock_info\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mBatchedSerializer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mCPickleSerializer\u001b[0m\u001b[1;33m(\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    819\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\py4j\\java_gateway.py\u001b[0m in \u001b[0;36m__call__\u001b[1;34m(self, *args)\u001b[0m\n\u001b[0;32m   1319\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1320\u001b[0m         \u001b[0manswer\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mself\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mgateway_client\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0msend_command\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcommand\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1321\u001b[1;33m         return_value = get_return_value(\n\u001b[0m\u001b[0;32m   1322\u001b[0m             answer, self.gateway_client, self.target_id, self.name)\n\u001b[0;32m   1323\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\Anaconda3\\lib\\site-packages\\pyspark\\sql\\utils.py\u001b[0m in \u001b[0;36mdeco\u001b[1;34m(*a, **kw)\u001b[0m\n\u001b[0;32m    194\u001b[0m                 \u001b[1;31m# Hide where the exception came from that shows a non-Pythonic\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    195\u001b[0m                 \u001b[1;31m# JVM exception message.\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m--> 196\u001b[1;33m                 \u001b[1;32mraise\u001b[0m \u001b[0mconverted\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m    197\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m    198\u001b[0m                 \u001b[1;32mraise\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mPythonException\u001b[0m: \n  An exception was thrown from the Python worker. Please see the stack trace below.\nTraceback (most recent call last):\n  File \"C:\\Users\\mmccoy\\Anaconda3\\Lib\\site-packages\\pyspark\\python\\lib\\pyspark.zip\\pyspark\\worker.py\", line 540, in main\nRuntimeError: Python in worker has different version 3.10 than that in driver 3.9, PySpark cannot run with different minor versions. Please check environment variables PYSPARK_PYTHON and PYSPARK_DRIVER_PYTHON are correctly set.\n"
     ]
    }
   ],
   "source": [
    "new_df = df_scaled.toPandas()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d7b54826",
   "metadata": {},
   "source": [
    "# Exploratory Analysis\n",
    "____________________________________________________________________________________________"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ff4031a5",
   "metadata": {},
   "source": [
    "Let's grab some descriptive statistics for scores."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0db7b473",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df.select('happiness_score').describe().show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "94065c08",
   "metadata": {},
   "outputs": [],
   "source": [
    "df.select(percentile_approx(\"happiness_score\", 0.25).alias(\"25th_percentile\"),\n",
    "                                percentile_approx(\"happiness_score\", 0.5).alias(\"50th_percentile\"),\n",
    "                                percentile_approx(\"happiness_score\", 0.75).alias(\"75th_percentile\")).show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f64e6758",
   "metadata": {},
   "outputs": [],
   "source": [
    "def percentile_val(df,col,perc):\n",
    "    df = df.select(percentile_approx(col,perc).alias(\"percentile_val\"))\n",
    "    return df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "688ab082",
   "metadata": {},
   "outputs": [],
   "source": [
    "(percentile_val(df, 'happiness_score', 0.5)).show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e1ce7fd5",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# happiness_df = df.select('country','happiness_score').where(col('happiness_score') > 5.559).toPandas()\n",
    "\n",
    "grouped_df = df.groupBy('country').agg(max('happiness_score').alias(\"desc_score\"))\n",
    "top_10_df = grouped_df.sort('desc_score', ascending=False).limit(10)\n",
    "top_10_df = top_10_df.toPandas()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "70711769",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(15, 6))\n",
    "ax.bar(top_10_df['country'], top_10_df['desc_score'], color='b')\n",
    "ax.set_title('Top 10 Countries by Happiness Score')\n",
    "ax.set_xlabel('Country')\n",
    "ax.set_ylabel('Happiness Score')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "726432c4",
   "metadata": {},
   "outputs": [],
   "source": [
    "num_df = df.drop('rank').drop('country')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6af955d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "num_df=num_df.toPandas()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9a7438af",
   "metadata": {},
   "outputs": [],
   "source": [
    "corr_df = num_df.corr()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "84ed7dbe",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, ax = plt.subplots(figsize=(15, 6))\n",
    "sns.heatmap(num_df.corr())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "00e99388",
   "metadata": {},
   "outputs": [],
   "source": [
    "corr_df_interesting = corr_df.where(abs(corr_df) > 0.7)\n",
    "display(corr_df_interesting)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "972f3a22",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_df = df.select('country','happiness_score').toPandas()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eda33efc",
   "metadata": {},
   "outputs": [],
   "source": [
    "type(test_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f3055df6",
   "metadata": {},
   "outputs": [],
   "source": [
    "import geopandas as gpd\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "# Load the world map shapefile\n",
    "world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))\n",
    "\n",
    "# Create a list of countries\n",
    "# countries = ['United States', 'Canada', 'Mexico', 'Brazil', 'Argentina']\n",
    "\n",
    "# Generate artificial happiness scores for each country\n",
    "# scores = pd.DataFrame({'Country': countries, 'Happiness Score': np.random.rand(len(countries))})\n",
    "\n",
    "scores = test_df = df.select('country','happiness_score').toPandas()\n",
    "\n",
    "# Merge the world map with the happiness scores data\n",
    "world = world.merge(scores, left_on='name', right_on='country')\n",
    "\n",
    "# Define the color map for the happiness scores\n",
    "cmap = 'Reds'\n",
    "\n",
    "# Plot the map with happiness scores as colors\n",
    "fig, ax = plt.subplots(figsize=(15,25))\n",
    "ax.set_aspect('equal')\n",
    "world.plot(\n",
    "    ax=ax,\n",
    "    column='happiness_score',\n",
    "    cmap=cmap\n",
    ")\n",
    "# Create a separate axis for the colorbar\n",
    "cax = fig.add_axes([1, 0.35, 0.05, 0.3])\n",
    "sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=world['happiness_score'].min(), vmax=world['happiness_score'].max()))\n",
    "sm._A = []\n",
    "fig.colorbar(sm, cax=cax)\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f4f6d0e5",
   "metadata": {},
   "source": [
    "Notice this map isn't super useful. Let's take a look an alternative map. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "58d50377",
   "metadata": {},
   "outputs": [],
   "source": [
    "#convert country column to a list\n",
    "list_ex = test_df['country'].tolist()\n",
    "list_ex = [c.replace('*','') for c in list_ex]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "347a11ab",
   "metadata": {},
   "outputs": [],
   "source": [
    "# List of all countries\n",
    "all_countries = list_ex\n",
    "\n",
    "# List of countries to remove - these are countries that don't match the country dictionary\n",
    "countries_to_remove = ['Kosovo', 'North Cyprus', 'Palestinian Territories','Eswatini. Kingdom of']\n",
    "\n",
    "# Create a new list that contains only the countries that are not in the list of countries to remove\n",
    "countries = [country for country in all_countries if country not in countries_to_remove]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d01a62bc",
   "metadata": {},
   "outputs": [],
   "source": [
    "from geopy.geocoders import Nominatim\n",
    "\n",
    "# countries = list_ex\n",
    "geolocator = Nominatim(user_agent=\"my_app\",timeout = 100)\n",
    "\n",
    "#Get the locations\n",
    "locations = []\n",
    "for country in countries:\n",
    "    location = geolocator.geocode(country)\n",
    "    code = country_name_to_country_alpha2(country, cn_name_format=\"default\")\n",
    "    cn_continent = country_alpha2_to_continent_code(code)\n",
    "    if location is not None:\n",
    "        locations.append((country, location.latitude, location.longitude, code, cn_continent))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b37c58b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "locations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "da33b836",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Create location dataframe\n",
    "locations_df = pd.DataFrame(locations, columns = ['country', 'latitude', 'longitude', 'code', 'continent'])\n",
    "locations_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "91f01c53",
   "metadata": {},
   "outputs": [],
   "source": [
    "#merge on country\n",
    "new_df = pd.merge(test_df, locations_df, on ='country', how ='inner')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "da4ffda9",
   "metadata": {},
   "outputs": [],
   "source": [
    "new_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d58180a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import folium\n",
    "from folium.plugins import MarkerCluster"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8072505f",
   "metadata": {},
   "outputs": [],
   "source": [
    "world_map = folium.Map(tile = \"cartodbpositron\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2c6a6adb",
   "metadata": {},
   "outputs": [],
   "source": [
    "marker_cluster = MarkerCluster().add_to(world_map)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8f7b1597",
   "metadata": {
    "scrolled": false
   },
   "outputs": [],
   "source": [
    "#for each coordinate, create circlemarker of user percent\n",
    "for i in range(len(new_df)):\n",
    "        lat = new_df.iloc[i]['latitude']\n",
    "        long = new_df.iloc[i]['longitude']\n",
    "        radius=5\n",
    "        popup_text = \"\"\"Country : {}<br>\n",
    "                    Happiness Score : {}<br>\"\"\"\n",
    "        popup_text = popup_text.format(new_df.iloc[i]['country'],\n",
    "                                   new_df.iloc[i]['happiness_score']\n",
    "                                   )\n",
    "        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True).add_to(marker_cluster)\n",
    "#show the map\n",
    "world_map.save('map.html')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d4f26a5a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
