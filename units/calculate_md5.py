import hashlib


async def calculate_md5(file):
    chunk_size = 4096
    md5 = hashlib.md5()
    for i in range(0, len(file), 4096):
        md5.update(file[i:i + chunk_size])
    return md5.hexdigest()
    # return hashlib.md5(file).hexdigest()
