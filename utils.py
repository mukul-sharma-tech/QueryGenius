import pandas as pd
from pymongo import MongoClient
import os
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, OperationFailure, BulkWriteError
import streamlit as st
from typing import Optional, List, Dict, Any, Union
import json
from urllib.parse import quote_plus
import time
from datetime import datetime
import numpy as np
from bson import ObjectId
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

def convert_datetime_for_json(data):
    """
    Recursively convert datetime objects to ISO strings for JSON serialization.
    """
    if isinstance(data, dict):
        return {key: convert_datetime_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_datetime_for_json(item) for item in data]
    elif isinstance(data, datetime):
        return data.isoformat()
    else:
        return data

def get_mongo_client() -> Optional[MongoClient]:
    """
    Create and return MongoDB Atlas client with proper configuration.
    Returns None if connection fails.
    """
    mongodb_uri = os.getenv("MONGODB_URI")
    
    if not mongodb_uri:
        st.error("❌ **MONGODB_URI not found in environment variables**")
        st.info("""
        📝 **Please create a .env file with:**
        ```
        MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
        GOOGLE_API_KEY=your_google_api_key
        ```
        """)
        return None
    
    try:
        # Fix password encoding issues
        if "@" in mongodb_uri:
            parts = mongodb_uri.split('@')
            if len(parts) == 2:
                auth_part = parts[0]
                if '://' in auth_part:
                    protocol, credentials = auth_part.split('://', 1)
                    if ':' in credentials:
                        username, password = credentials.split(':', 1)
                        # URL encode special characters in password
                        encoded_password = quote_plus(password)
                        mongodb_uri = f"{protocol}://{username}:{encoded_password}@{parts[1]}"
        
        st.info(f"🔗 Connecting to MongoDB...")
        
        client = MongoClient(
            mongodb_uri,
            serverSelectionTimeoutMS=20000,
            connectTimeoutMS=30000,
            socketTimeoutMS=45000,
            retryWrites=True,
            maxPoolSize=100,
            connect=True
        )
        
        # Test connection with timeout
        start_time = time.time()
        client.admin.command('ping')
        connection_time = time.time() - start_time
        
        st.success(f"✅ **Connected to MongoDB!** ({(connection_time*1000):.0f}ms)")
        return client
        
    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ **MongoDB Connection Failed:** {error_msg[:200]}")
        
        # Diagnostic messages
        if "bad auth" in error_msg.lower() or "authentication" in error_msg.lower():
            st.info("""
            🔐 **AUTHENTICATION ERROR:**
            1. **Check credentials** in your connection string
            2. **Create database user** in MongoDB Atlas:
               - Go to: Atlas → Security → Database Access
               - Click: "Add New Database User"
               - Authentication Method: "Password"
               - Set username & password
               - Privileges: "Read and write to any database"
            3. **Wait 1-2 minutes** after creating user
            """)
        elif "timed out" in error_msg.lower():
            st.info("""
            ⏱️ **CONNECTION TIMEOUT:**
            1. **Check internet connection**
            2. **Whitelist your IP** in MongoDB Atlas:
               - Go to: Atlas → Security → Network Access
               - Click: "Add IP Address"
               - Add: `0.0.0.0/0` (temporarily for testing)
               - Click: "Confirm"
            3. **Try again** after 1 minute
            """)
        elif "invalid" in error_msg.lower() or "uri" in error_msg.lower():
            st.info("""
            🔗 **INVALID CONNECTION STRING:**
            1. **Get fresh connection string:**
               - Login to MongoDB Atlas
               - Click your cluster → "Connect"
               - Choose "Connect your application"
               - Copy the connection string
            2. **Format should be:**
               ```
               mongodb+srv://username:password@cluster.mongodb.net/
               ```
            """)
        elif "network error" in error_msg.lower():
            st.info("""
            🌐 **NETWORK ERROR:**
            1. **Check firewall settings**
            2. **Try different network** (mobile hotspot)
            3. **Contact your network administrator**
            """)
        else:
            st.info("""
            ⚠️ **GENERAL TROUBLESHOOTING:**
            1. **Check if cluster is running** in MongoDB Atlas
            2. **Ensure you have sufficient credits** (free tier has limits)
            3. **Try re-creating the cluster**
            """)
        
        return None
def upload_to_mongo(file, collection_name: str = "dataset") -> int:
    """
    Upload CSV/Excel file to MongoDB Atlas.
    Returns number of records uploaded.
    """
    if file is None:
        st.error("❌ No file provided")
        return 0
    
    try:
        client = get_mongo_client()
        if client is None:
            st.error("❌ Cannot connect to MongoDB")
            return 0
        
        db = client["nl_mongo_db"]
        
        # Read file based on extension
        st.info(f"📄 Reading {file.name}...")
        
        # Save file to temp location for debugging
        import tempfile
        import shutil
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            shutil.copyfileobj(file, tmp_file)
            tmp_file_path = tmp_file.name
        
        # Reset file pointer
        file.seek(0)
        
        df = None
        file_ext = file.name.lower()
        
        try:
            if file_ext.endswith(".csv"):
                # Try multiple CSV reading strategies
                try:
                    # Strategy 1: Default reading
                    df = pd.read_csv(file)
                except pd.errors.EmptyDataError:
                    st.error("❌ CSV file is completely empty")
                    return 0
                except pd.errors.ParserError as e:
                    st.warning(f"⚠️ First CSV parse failed: {str(e)[:100]}")
                    
                    # Strategy 2: Try with different parameters
                    try:
                        file.seek(0)
                        df = pd.read_csv(file, encoding='utf-8')
                    except:
                        try:
                            file.seek(0)
                            df = pd.read_csv(file, encoding='latin-1')
                        except:
                            try:
                                file.seek(0)
                                df = pd.read_csv(file, encoding='ISO-8859-1')
                            except:
                                # Strategy 3: Try with error handling
                                file.seek(0)
                                df = pd.read_csv(file, engine='python', on_bad_lines='skip')
                
                # If still None, try reading the temp file
                if df is None or df.empty:
                    try:
                        df = pd.read_csv(tmp_file_path, engine='python', on_bad_lines='skip')
                    except Exception as e:
                        st.error(f"❌ Failed to read CSV: {str(e)}")
                        
                        # Show file content for debugging
                        file.seek(0)
                        content = file.read(1000).decode('utf-8', errors='ignore')
                        st.code(f"First 1000 chars of file:\n{content}", language="text")
                        return 0
                        
            elif file_ext.endswith(".xlsx"):
                df = pd.read_excel(file, engine='openpyxl')
            elif file_ext.endswith(".xls"):
                df = pd.read_excel(file, engine='xlrd')
            else:
                st.error(f"❌ Unsupported file format: {file.name}")
                os.unlink(tmp_file_path)
                return 0
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            
            # Clean up temp file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            
            # Provide troubleshooting tips
            st.info("""
            🔧 **Troubleshooting file reading:**
            
            1. **Check file format:** Ensure it's a valid CSV/Excel file
            2. **Check encoding:** Try saving as UTF-8 CSV
            3. **Check content:** File should have column headers and data
            4. **Check file size:** File should not be empty
            5. **Try re-saving:** Open in Excel and save as CSV UTF-8
            
            **For CSV files:**
            - First line should be column headers
            - Use commas as separators (not semicolons)
            - Avoid special characters in headers
            """)
            return 0
        
        # Clean up temp file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        
        # Validate dataframe
        if df is None or df.empty:
            st.warning("⚠️ The uploaded file is empty or contains no data")
            return 0
        
        # Check if dataframe has columns
        if len(df.columns) == 0:
            st.error("❌ No columns found in the file")
            st.info("""
            The file appears to have no column headers. 
            CSV files should have column names in the first row.
            
            Example:
            ```
            name,email,marks,salary,job_location
            John,john@email.com,85,50000,Delhi
            ```
            """)
            return 0
        
        st.success(f"✅ Read {len(df):,} rows with {len(df.columns)} columns")
        
        # Show column info
        with st.expander("📋 Column Details", expanded=True):
            col_info = pd.DataFrame({
                'Column Name': df.columns,
                'Data Type': df.dtypes.astype(str).values,
                'Non-Null Count': df.notnull().sum().values,
                'Sample Value': df.iloc[0].values if len(df) > 0 else [''] * len(df.columns)
            })
            st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        # Clean column names (remove spaces, special chars)
        original_cols = df.columns.tolist()
        df.columns = [str(col).strip().replace(' ', '_').replace('.', '_').replace('(', '').replace(')', '').lower() 
                     for col in df.columns]
        
        if original_cols != df.columns.tolist():
            st.info(f"📝 Column names cleaned")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original:**")
                for col in original_cols[:5]:
                    st.write(f"- {col}")
            with col2:
                st.write("**Cleaned:**")
                for col in df.columns[:5]:
                    st.write(f"- {col}")
        
        # Convert to records and handle NaN values
        records = df.replace({np.nan: None}).to_dict(orient="records")
        
        # Create or get collection
        col = db[collection_name]
        
        # Clear old data if exists
        try:
            existing_count = col.count_documents({})
            if existing_count > 0:
                with st.spinner(f"🗑️ Clearing existing data ({existing_count:,} records)..."):
                    col.delete_many({})
                st.info(f"🗑️ Cleared {existing_count:,} existing records from '{collection_name}'")
        except Exception as e:
            st.warning(f"⚠️ Could not clear existing data: {str(e)[:100]}")
        
        # Insert new data in batches
        if records:
            batch_size = 500  # Reduced for better progress tracking
            total_inserted = 0
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                try:
                    result = col.insert_many(batch, ordered=False)
                    total_inserted += len(result.inserted_ids)
                    
                    # Update progress
                    progress = min((i + len(batch)) / len(records), 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"📤 Uploading: {total_inserted:,} / {len(records):,} records")
                    
                except BulkWriteError as bwe:
                    # Handle partial failures
                    inserted = bwe.details.get('nInserted', 0)
                    total_inserted += inserted
                    errors = bwe.details.get('writeErrors', [])
                    if errors:
                        st.warning(f"⚠️ Some records failed. Successfully inserted {inserted} records")
                        
                except Exception as e:
                    st.warning(f"⚠️ Batch {i//batch_size + 1} failed: {str(e)[:100]}")
                    # Try inserting records one by one
                    successful_in_batch = 0
                    for record in batch:
                        try:
                            col.insert_one(record)
                            successful_in_batch += 1
                            total_inserted += 1
                        except:
                            continue
                    
                    if successful_in_batch > 0:
                        st.info(f"Inserted {successful_in_batch} records from failed batch")
            
            progress_bar.empty()
            status_text.empty()
            
            if total_inserted > 0:
                st.success(f"✅ Successfully uploaded **{total_inserted:,}** records to '{collection_name}'")
                
                # Show sample of uploaded data
                with st.expander("👀 Sample of Uploaded Data", expanded=False):
                    sample_data = list(col.find({}, {"_id": 0}).limit(5))
                    if sample_data:
                        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)
                
                # Show collection info
                try:
                    col_stats = db.command("collstats", collection_name)
                    st.info(f"""
                    📊 **Collection Stats:**
                    - **Documents:** {col_stats.get('count', 0):,}
                    - **Size:** {col_stats.get('size', 0) / (1024*1024):.2f} MB
                    - **Storage:** {col_stats.get('storageSize', 0) / (1024*1024):.2f} MB
                    - **Avg Document Size:** {col_stats.get('avgObjSize', 0):.0f} bytes
                    """)
                except:
                    st.info(f"📊 **Upload Summary:** {total_inserted:,} documents uploaded")
                
                return total_inserted
            else:
                st.error("❌ No records were inserted")
                return 0
        else:
            st.warning("⚠️ No valid records to insert")
            return 0
            
    except Exception as e:
        st.error(f"❌ Unexpected error during upload: {str(e)}")
        import traceback
        st.code(traceback.format_exc(), language="python")
        return 0
        
    finally:
        try:
            if 'client' in locals() and client:
                client.close()
        except:
            pass
def fetch_schema(collection_name: str = "dataset") -> Dict[str, Any]:
    """
    Return first document for schema display.
    """
    client = get_mongo_client()
    if client is None:
        return {"error": "No database connection"}
    
    db = client["nl_mongo_db"]
    
    try:
        col = db[collection_name]
        
        # Check if collection exists
        if collection_name not in db.list_collection_names():
            return {"error": f"Collection '{collection_name}' not found"}
        
        # Check if collection has data
        count = col.count_documents({})
        if count == 0:
            return {"message": "Collection is empty", "count": 0}
        
        # Get sample document(s) to understand schema
        sample = col.find_one({}, {"_id": 0})

        if not sample:
            return {"message": "No documents found", "count": count}

        # Convert datetime objects in sample document for JSON serialization
        processed_sample = convert_datetime_for_json(sample)

        # Get field types from multiple samples
        samples = list(col.find({}, {"_id": 0}).limit(10))
        field_info = {}

        for doc in samples:
            for key, value in doc.items():
                if key not in field_info:
                    field_info[key] = {
                        "type": type(value).__name__,
                        "sample_values": set(),
                        "count": 0
                    }

                field_info[key]["sample_values"].add(str(value)[:50])
                field_info[key]["count"] += 1

        # Convert sets to lists for JSON serialization
        for key in field_info:
            field_info[key]["sample_values"] = list(field_info[key]["sample_values"])[:5]

        schema_info = {
            "collection": collection_name,
            "count": count,
            "sample_document": processed_sample,
            "fields": list(field_info.keys()),
            "field_details": field_info,
            "timestamp": datetime.now().isoformat()
        }
        
        return schema_info
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            if client:
                client.close()
        except:
            pass

def execute_mongo_query(query_pipeline: List[Dict], collection_name: str = "dataset") -> List[Dict]:
    """
    Run aggregation pipeline and return results.
    """
    client = get_mongo_client()
    if client is None:
        return []
    
    db = client["nl_mongo_db"]
    
    try:
        # Check if collection exists
        if collection_name not in db.list_collection_names():
            st.error(f"❌ Collection '{collection_name}' not found")
            return []
        
        col = db[collection_name]
        
        # Check if collection has data
        if col.count_documents({}) == 0:
            st.warning(f"⚠️ Collection '{collection_name}' is empty")
            return []
        
        # Execute aggregation pipeline
        start_time = time.time()
        result = list(col.aggregate(query_pipeline))
        execution_time = time.time() - start_time
        
        if result:
            st.success(f"✅ Query executed in {execution_time*1000:.0f}ms, returned {len(result)} records")
        
        # Convert all datetime objects and ObjectIds for JSON serialization
        result = convert_datetime_for_json(result)
        for doc in result:
            if '_id' in doc and isinstance(doc['_id'], ObjectId):
                doc['_id'] = str(doc['_id'])
        
        return result
        
    except Exception as e:
        st.error(f"❌ Query execution error: {str(e)[:200]}")
        return []
    finally:
        try:
            if client:
                client.close()
        except:
            pass

def get_all_collections() -> List[str]:
    """
    Get all collection names from the database.
    """
    client = get_mongo_client()
    if client is None:
        return []
    
    db = client["nl_mongo_db"]
    
    try:
        collections = db.list_collection_names()
        return sorted(collections)
    except Exception as e:
        st.error(f"❌ Error fetching collections: {str(e)[:100]}")
        return []
    finally:
        try:
            if client:
                client.close()
        except:
            pass

def get_collection_stats(collection_name: str = "dataset") -> Dict[str, Any]:
    """
    Get statistics for a collection.
    """
    client = get_mongo_client()
    if client is None:
        return {}
    
    db = client["nl_mongo_db"]
    
    try:
        if collection_name not in db.list_collection_names():
            return {"error": f"Collection '{collection_name}' not found"}
        
        col = db[collection_name]
        
        # Get basic stats
        stats = {
            "name": collection_name,
            "count": col.count_documents({}),
            "indexes": len(col.index_information()),
            "fields": [],
            "created": None,
            "size_mb": 0,
            "storage_mb": 0
        }
        
        # Try to get collection stats
        try:
            coll_stats = db.command("collstats", collection_name)
            stats.update({
                "size_mb": coll_stats.get("size", 0) / (1024 * 1024),
                "storage_mb": coll_stats.get("storageSize", 0) / (1024 * 1024),
                "avg_obj_size": coll_stats.get("avgObjSize", 0),
                "nindexes": coll_stats.get("nindexes", 0),
                "total_index_size_mb": coll_stats.get("totalIndexSize", 0) / (1024 * 1024)
            })
        except:
            pass
        
        # Get sample document for fields
        sample = col.find_one({}, {"_id": 0})
        if sample:
            stats["fields"] = list(sample.keys())
            
            # Try to infer data types
            field_types = {}
            for key, value in sample.items():
                field_types[key] = type(value).__name__
            stats["field_types"] = field_types
        
        return stats
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            if client:
                client.close()
        except:
            pass

def get_collection_data(collection_name: str = "dataset", limit: int = 1000, skip: int = 0) -> pd.DataFrame:
    """
    Get data from a collection as pandas DataFrame.
    """
    client = get_mongo_client()
    if client is None:
        return pd.DataFrame()
    
    db = client["nl_mongo_db"]
    
    try:
        if collection_name not in db.list_collection_names():
            return pd.DataFrame()
        
        col = db[collection_name]
        
        # Get data with limit and skip
        cursor = col.find({}, {"_id": 0}).skip(skip).limit(limit)
        data = list(cursor)
        
        if data:
            df = pd.DataFrame(data)
            return df
        else:
            return pd.DataFrame()
            
    except Exception:
        return pd.DataFrame()
    finally:
        try:
            if client:
                client.close()
        except:
            pass

def delete_collection(collection_name: str) -> bool:
    """
    Delete a collection from the database.
    """
    client = get_mongo_client()
    if client is None:
        return False
    
    db = client["nl_mongo_db"]
    
    try:
        if collection_name not in db.list_collection_names():
            st.error(f"❌ Collection '{collection_name}' not found")
            return False
        
        # Get count before deletion
        col = db[collection_name]
        count = col.count_documents({})
        
        # Drop collection
        db.drop_collection(collection_name)
        
        st.success(f"✅ Deleted collection '{collection_name}' with {count:,} records")
        return True
        
    except Exception as e:
        st.error(f"❌ Error deleting collection: {str(e)}")
        return False
    finally:
        try:
            if client:
                client.close()
        except:
            pass

def search_in_collection(collection_name: str, search_text: str, field: str = None) -> List[Dict]:
    """
    Search for text in a collection using regex.
    """
    client = get_mongo_client()
    if client is None:
        return []
    
    db = client["nl_mongo_db"]
    
    try:
        if collection_name not in db.list_collection_names():
            return []
        
        col = db[collection_name]
        
        # Create search query
        if field:
            # Search in specific field
            query = {field: {"$regex": search_text, "$options": "i"}}
        else:
            # Search in all string fields
            sample_doc = col.find_one({}, {"_id": 0})
            if not sample_doc:
                return []
            
            string_fields = [k for k, v in sample_doc.items() if isinstance(v, str)]
            if not string_fields:
                return []
            
            # Create OR query for all string fields
            or_conditions = [{f: {"$regex": search_text, "$options": "i"}} for f in string_fields]
            query = {"$or": or_conditions} if len(or_conditions) > 1 else or_conditions[0]
        
        # Execute search
        results = list(col.find(query, {"_id": 0}).limit(100))
        return results
        
    except Exception:
        return []
    finally:
        try:
            if client:
                client.close()
        except:
            pass

def get_collection_aggregations(collection_name: str) -> Dict[str, Any]:
    """
    Get common aggregations for a collection.
    """
    client = get_mongo_client()
    if client is None:
        return {}
    
    db = client["nl_mongo_db"]
    
    try:
        if collection_name not in db.list_collection_names():
            return {}
        
        col = db[collection_name]
        
        # Get sample document to identify numeric fields
        sample = col.find_one({}, {"_id": 0})
        if not sample:
            return {}
        
        numeric_fields = [k for k, v in sample.items() if isinstance(v, (int, float))]
        aggregations = {"numeric_fields": numeric_fields}
        
        # Calculate basic stats for each numeric field
        for field in numeric_fields:
            try:
                pipeline = [
                    {
                        "$group": {
                            "_id": None,
                            f"avg_{field}": {"$avg": f"${field}"},
                            f"min_{field}": {"$min": f"${field}"},
                            f"max_{field}": {"$max": f"${field}"},
                            f"sum_{field}": {"$sum": f"${field}"},
                            f"std_{field}": {"$stdDevSamp": f"${field}"}
                        }
                    }
                ]
                
                result = list(col.aggregate(pipeline))
                if result:
                    aggregations[field] = result[0]
            except:
                continue
        
        # Get value counts for string fields (top 10)
        string_fields = [k for k, v in sample.items() if isinstance(v, str)]
        for field in string_fields[:5]:  # Limit to 5 fields to avoid performance issues
            try:
                pipeline = [
                    {"$group": {"_id": f"${field}", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": 10}
                ]
                result = list(col.aggregate(pipeline))
                aggregations[f"top_{field}"] = result
            except:
                continue
        
        return aggregations
        
    except Exception:
        return {}
    finally:
        try:
            if client:
                client.close()
        except:
            pass

def test_mongo_connection() -> Dict[str, Any]:
    """
    Test MongoDB connection and return detailed status.
    """
    start_time = time.time()
    client = get_mongo_client()
    
    if client is None:
        return {
            "connected": False,
            "error": "Failed to create client. Check MONGODB_URI in .env file.",
            "response_time": time.time() - start_time
        }
    
    try:
        # Test ping
        ping_result = client.admin.command('ping')
        
        # Get database info
        db = client["nl_mongo_db"]
        collections = db.list_collection_names()
        
        # Get server info
        server_info = client.server_info()
        
        result = {
            "connected": True,
            "database": "nl_mongo_db",
            "collections_count": len(collections),
            "collections": collections,
            "server_version": server_info.get("version", "Unknown"),
            "host": str(client.address),
            "response_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        return {
            "connected": False,
            "error": str(e),
            "response_time": time.time() - start_time
        }
    finally:
        try:
            if client:
                client.close()
        except:
            pass

def get_database_info() -> Dict[str, Any]:
    """
    Get comprehensive database information.
    """
    client = get_mongo_client()
    if client is None:
        return {"error": "No connection"}
    
    db = client["nl_mongo_db"]
    
    try:
        collections = db.list_collection_names()
        info = {
            "database": "nl_mongo_db",
            "total_collections": len(collections),
            "collections": [],
            "total_documents": 0,
            "total_size_mb": 0,
            "server_info": client.server_info()
        }
        
        # Get info for each collection
        for collection in collections:
            col = db[collection]
            try:
                coll_stats = db.command("collstats", collection)
                stats = {
                    "name": collection,
                    "count": coll_stats.get("count", 0),
                    "size_mb": coll_stats.get("size", 0) / (1024 * 1024),
                    "storage_mb": coll_stats.get("storageSize", 0) / (1024 * 1024),
                    "avg_obj_size": coll_stats.get("avgObjSize", 0),
                    "indexes": coll_stats.get("nindexes", 0),
                    "total_index_size_mb": coll_stats.get("totalIndexSize", 0) / (1024 * 1024)
                }
                
                # Add to totals
                info["total_documents"] += stats["count"]
                info["total_size_mb"] += stats["size_mb"]
                
                info["collections"].append(stats)
            except:
                # Fallback if collstats fails
                stats = {
                    "name": collection,
                    "count": col.count_documents({}),
                    "size_mb": 0,
                    "indexes": len(col.index_information())
                }
                info["collections"].append(stats)
        
        return info
        
    except Exception as e:
        return {"error": str(e)}
    finally:
        try:
            if client:
                client.close()
        except:
            pass

def get_field_distinct_values(collection_name: str, field_name: str, limit: int = 100) -> List:
    """
    Get distinct values for a specific field.
    """
    client = get_mongo_client()
    if client is None:
        return []
    
    db = client["nl_mongo_db"]
    
    try:
        if collection_name not in db.list_collection_names():
            return []
        
        col = db[collection_name]
        values = col.distinct(field_name)
        return list(values)[:limit]
        
    except Exception:
        return []
    finally:
        try:
            if client:
                client.close()
        except:
            pass

def export_collection_to_csv(collection_name: str) -> str:
    """
    Export collection data to CSV format.
    """
    df = get_collection_data(collection_name, limit=10000)
    
    if df.empty:
        return ""
    
    csv_content = df.to_csv(index=False)
    return csv_content

def create_index(collection_name: str, field: str, unique: bool = False) -> bool:
    """
    Create an index on a field.
    """
    client = get_mongo_client()
    if client is None:
        return False
    
    db = client["nl_mongo_db"]
    
    try:
        if collection_name not in db.list_collection_names():
            return False
        
        col = db[collection_name]
        col.create_index([(field, 1)], unique=unique)
        return True
        
    except Exception:
        return False
    finally:
        try:
            if client:
                client.close()
        except:
            pass

def backup_collection(collection_name: str) -> List[Dict]:
    """
    Backup collection data.
    """
    client = get_mongo_client()
    if client is None:
        return []
    
    db = client["nl_mongo_db"]
    
    try:
        if collection_name not in db.list_collection_names():
            return []
        
        col = db[collection_name]
        data = list(col.find({}, {"_id": 0}))
        return data
        
    except Exception:
        return []
    finally:
        try:
            if client:
                client.close()
        except:
            pass

def restore_collection(collection_name: str, data: List[Dict]) -> bool:
    """
    Restore collection data.
    """
    if not data:
        return False
    
    client = get_mongo_client()
    if client is None:
        return False
    
    db = client["nl_mongo_db"]
    
    try:
        col = db[collection_name]
        
        # Clear existing data
        col.delete_many({})
        
        # Insert backup data in batches
        batch_size = 1000
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            col.insert_many(batch)
        
        return True
        
    except Exception:
        return False
    finally:
        try:
            if client:
                client.close()
        except:
            pass

def get_query_examples(collection_name: str = "dataset") -> List[Dict[str, str]]:
    """
    Get query examples based on collection schema.
    """
    schema = fetch_schema(collection_name)
    
    if "error" in schema or "message" in schema:
        return []
    
    fields = schema.get("fields", [])
    examples = []
    
    # Basic queries
    examples.append({
        "query": "Show all data",
        "description": "Display all records",
        "pipeline": "[]"
    })
    
    # Sort queries
    for field in fields:
        examples.append({
            "query": f"Sort by {field} ascending",
            "description": f"Sort data by {field}",
            "pipeline": f'[{{"$sort": {{"{field}": 1}}}}]'
        })
        
        examples.append({
            "query": f"Sort by {field} descending",
            "description": f"Sort data by {field} in reverse",
            "pipeline": f'[{{"$sort": {{"{field}": -1}}}}]'
        })
    
    # Limit queries
    examples.append({
        "query": "Show first 10 records",
        "description": "Limit results",
        "pipeline": '[{"$limit": 10}]'
    })
    
    examples.append({
        "query": "Show top 5 records",
        "description": "Get top records",
        "pipeline": '[{"$limit": 5}]'
    })
    
    # Projection queries
    if len(fields) > 0:
        examples.append({
            "query": f"Show only {fields[0]} field",
            "description": "Select specific fields",
            "pipeline": f'[{{"$project": {{"{fields[0]}": 1, "_id": 0}}}}]'
        })
    
    return examples[:20]  # Limit to 20 examples

def validate_query_pipeline(pipeline: List[Dict]) -> Dict[str, Any]:
    """
    Validate a MongoDB aggregation pipeline.
    """
    if not isinstance(pipeline, list):
        return {"valid": False, "error": "Pipeline must be a list"}
    
    valid_stages = {
        "$match", "$group", "$sort", "$limit", "$skip", "$project",
        "$unwind", "$lookup", "$addFields", "$count", "$facet",
        "$bucket", "$sortByCount", "$graphLookup", "$search"
    }
    
    for i, stage in enumerate(pipeline):
        if not isinstance(stage, dict):
            return {"valid": False, "error": f"Stage {i} must be a dictionary"}
        
        if len(stage) != 1:
            return {"valid": False, "error": f"Stage {i} must have exactly one operator"}

        operator = list(stage.keys())[0]
        if operator not in valid_stages:
            return {"valid": False, "error": f"Invalid operator '{operator}' in stage {i}"}

    return {"valid": True, "message": "Pipeline is valid"}

def trend_prediction_model(data: pd.DataFrame, target_column: str, periods: int = 12) -> Dict[str, Any]:
    """
    Perform trend prediction using ARIMA model.
    """
    try:
        if data.empty or target_column not in data.columns:
            return {"error": "Invalid data or target column"}

        # Prepare time series data
        ts_data = data.copy()
        if 'date' in ts_data.columns:
            ts_data['date'] = pd.to_datetime(ts_data['date'], errors='coerce')
            ts_data = ts_data.dropna(subset=['date'])
            ts_data = ts_data.set_index('date').sort_index()

        # Aggregate by date if multiple entries
        if ts_data.index.duplicated().any():
            ts_data = ts_data.groupby(ts_data.index).sum()

        # Fit ARIMA model
        model = ARIMA(ts_data[target_column], order=(5,1,0))
        model_fit = model.fit()

        # Forecast
        forecast = model_fit.forecast(steps=periods)
        forecast_dates = pd.date_range(start=ts_data.index[-1], periods=periods+1, freq='D')[1:]

        # Convert historical data to ensure JSON serializable
        historical_dict = {}
        for idx, val in ts_data[target_column].tail(30).items():
            key = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
            historical_dict[key] = float(val)

        return {
            "historical": historical_dict,
            "forecast": dict(zip(forecast_dates.strftime('%Y-%m-%d'), forecast.values)),
            "model_summary": str(model_fit.summary())
        }

    except Exception as e:
        return {"error": str(e)}

def anomaly_detection(data: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
    """
    Detect anomalies using Isolation Forest.
    """
    try:
        if data.empty:
            return {"error": "No data provided"}

        # Select numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {"error": "No numeric columns found"}

        # Prepare data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data[numeric_cols])

        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomalies = iso_forest.fit_predict(scaled_data)

        # Get anomaly scores
        scores = iso_forest.decision_function(scaled_data)

        # Create results
        results = data.copy()
        results['anomaly_score'] = scores
        results['is_anomaly'] = anomalies == -1

        anomaly_count = results['is_anomaly'].sum()
        total_count = len(results)

        # Convert anomalous records to dict, handling datetime objects
        anomalous_df = results[results['is_anomaly']].head(10)
        anomalous_records = []
        for _, row in anomalous_df.iterrows():
            record = convert_datetime_for_json(row.to_dict())
            anomalous_records.append(record)

        return {
            "anomaly_count": int(anomaly_count),
            "total_records": total_count,
            "anomaly_percentage": (anomaly_count / total_count) * 100,
            "anomalous_records": anomalous_records,
            "contamination_used": contamination
        }

    except Exception as e:
        return {"error": str(e)}

def correlation_analysis(data1: pd.DataFrame, data2: pd.DataFrame, method: str = 'pearson') -> Dict[str, Any]:
    """
    Perform correlation analysis between two datasets.
    """
    try:
        if data1.empty or data2.empty:
            return {"error": "One or both datasets are empty"}

        # Find common columns
        common_cols = set(data1.columns) & set(data2.columns)
        numeric_common = [col for col in common_cols if pd.api.types.is_numeric_dtype(data1[col]) and pd.api.types.is_numeric_dtype(data2[col])]

        if not numeric_common:
            return {"error": "No common numeric columns found"}

        correlations = {}
        for col in numeric_common:
            corr = data1[col].corr(data2[col], method=method)
            correlations[col] = corr

        # Overall correlation matrix
        combined_data = pd.concat([data1[numeric_common], data2[numeric_common]], axis=1, keys=['dataset1', 'dataset2'])
        corr_matrix = combined_data.corr(method=method)

        return {
            "column_correlations": correlations,
            "correlation_matrix": corr_matrix.to_dict(),
            "strongest_correlations": sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        }

    except Exception as e:
        return {"error": str(e)}

def generate_ml_report(trend_results: Dict, anomaly_results: Dict, correlation_results: Dict) -> str:
    """
    Generate automated ML report.
    """
    try:
        report = f"""
# UIDAI Data Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Trend Prediction Analysis
"""

        if 'error' not in trend_results:
            report += f"""
- Historical data points analyzed: {len(trend_results.get('historical', {}))}
- Forecast periods: {len(trend_results.get('forecast', {}))}
- Model: ARIMA(5,1,0)
"""
        else:
            report += f"- Error: {trend_results['error']}\n"

        report += "\n## Anomaly Detection Results\n"
        if 'error' not in anomaly_results:
            report += f"""
- Total records analyzed: {anomaly_results.get('total_records', 0)}
- Anomalies detected: {anomaly_results.get('anomaly_count', 0)}
- Anomaly percentage: {anomaly_results.get('anomaly_percentage', 0):.2f}%
- Contamination parameter: {anomaly_results.get('contamination_used', 0)}
"""
        else:
            report += f"- Error: {anomaly_results['error']}\n"

        report += "\n## Correlation Analysis\n"
        if 'error' not in correlation_results:
            strongest = correlation_results.get('strongest_correlations', [])
            report += f"""
- Columns analyzed: {len(correlation_results.get('column_correlations', {}))}
- Strongest correlations:
"""
            for col, corr in strongest:
                report += f"  - {col}: {corr:.3f}\n"
        else:
            report += f"- Error: {correlation_results['error']}\n"

        report += "\n## Recommendations\n"
        report += "- Monitor trends for policy planning\n"
        report += "- Investigate detected anomalies\n"
        report += "- Use correlation insights for targeted interventions\n"

        return report

    except Exception as e:
        return f"Error generating report: {str(e)}"