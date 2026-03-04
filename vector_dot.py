# data: emb和emb2是字符串格式的向量

# 1. 清洗并解析为浮点数数组
df_parsed = data.withColumn(
    "vec_a_clean", F.regexp_replace("emb", "\\s", "")   # 去除空格
).withColumn(
    "vec_b_clean", F.regexp_replace("emb2", "\\s", "")
).withColumn(
    "vec_a", F.split("vec_a_clean", ",").cast("array<float>")
).withColumn(
    "vec_b", F.split("vec_b_clean", ",").cast("array<float>")
)

# 2. 计算点积（假设向量长度相同，若不同可加长度检查）
df_with_dot = df_parsed.withColumn(
    "dot_product",
    F.aggregate(
        F.zip_with("vec_a", "vec_b", lambda x, y: x * y),   # 对应元素相乘
        F.lit(0.0),                                          # 初始值
        lambda acc, z: acc + z                               # 累加求和
    )
)
df_with_dot = df_with_dot.withColumn('dot_product_rounded', F.round("dot_product", 4))
df_with_dot = df_with_dot.select('cluster_code', 'job_id', 'job_id2', 'emb', 'emb2', 'dot_product_rounded').distinct()


# 3. 按点积降序排序，取前200
window_spec = Window.partitionBy("cluster_code", "job_id").orderBy(F.col("dot_product_rounded").desc())
df_ranked = df_with_dot.withColumn("rn", F.row_number().over(window_spec))
top200_per_group = df_ranked.filter(F.col("rn") <= 200)
top200_per_group.show(20, False)


# 4. 按照顺序拼接相似的job和相似分（点积）
final_res = top200_per_group.withColumn('sim_job_score', F.concat_ws(':', 'job_id2', 'dot_product_rounded'))
final_res = final_res.groupBy('cluster_code', 'job_id').agg(F.concat_ws(",", F.collect_set("sim_job_score")).alias("target_nodes"))
final_res = final_res.withColumn('source_node', F.col('job_id')).withColumn('ds', F.lit(today)).withColumn('source', F.lit('sim_job'))

