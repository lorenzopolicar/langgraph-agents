from sql_agent.subgraph import download_db

db = download_db()

"""
SELECT
    TABLE_NAME,
    TABLE_COMMENT
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'cms' and table_comment regexp "Table description";
"""
result = db.run_no_throw(
    "SELECT TABLE_NAME, TABLE_COMMENT FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = 'cms'"
)


print(result)
