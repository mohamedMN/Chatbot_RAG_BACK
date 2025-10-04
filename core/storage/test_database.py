import os
import json
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

url = os.environ.get("supabaseUrl")
key = os.environ.get("supabaseKey")
bucket = (os.environ.get("SUPABASE_BUCKET") or "rag-artifacts").strip()

print("URL:", url)
print("Key prefix:", (key or "")[:12])
print("Bucket:", bucket)

sb = create_client(url, key)

# List buckets (works with Service Role key)
try:
    names = [getattr(b, "name", getattr(b, "id", None))
             for b in sb.storage.list_buckets()]
    print("Buckets in this project:", names)
except Exception as e:
    print("list_buckets failed:", e)

# Try an upload
try:
    data = json.dumps({"ok": True}).encode("utf-8")
    sb.storage.from_(bucket).upload("health/test.json", data,
                                    {"content-type": "application/json", "upsert": "true"})
    url = sb.storage.from_(bucket).get_public_url("health/test.json")
    print("Upload OK. Public URL:", url)
except Exception as e:
    print("Upload failed:", e)
