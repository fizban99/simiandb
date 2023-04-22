# -*- coding: utf-8 -*-
import tables
from pathlib import Path
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from tqdm import tqdm
from numba import njit, prange
from time import time



@njit('float32[:](uint8[:])', parallel=True)
def tofp32n8(arr):
    """Numba-optimized function that converts a fp8 (4M3E) array to fp32 using a mapping table
    The array is assumed to be one dimensional with the fp8
    represented as UInt8
    """
    fp8table= np.frombuffer(b'\x00\x00\x00\x00\x00\x00\x00;\x00\x00\x80;\x00\x00\xc0;\x00\x00\x00<\x00\x00 <\x00\x00@<\x00\x00`<\x00\x00\x80<\x00\x00\x90<\x00\x00\xa0<\x00\x00\xb0<\x00\x00\xc0<\x00\x00\xd0<\x00\x00\xe0<\x00\x00\xf0<\x00\x00\x00=\x00\x00\x10=\x00\x00 =\x00\x000=\x00\x00@=\x00\x00P=\x00\x00`=\x00\x00p=\x00\x00\x80=\x00\x00\x90=\x00\x00\xa0=\x00\x00\xb0=\x00\x00\xc0=\x00\x00\xd0=\x00\x00\xe0=\x00\x00\xf0=\x00\x00\x00>\x00\x00\x10>\x00\x00 >\x00\x000>\x00\x00@>\x00\x00P>\x00\x00`>\x00\x00p>\x00\x00\x80>\x00\x00\x90>\x00\x00\xa0>\x00\x00\xb0>\x00\x00\xc0>\x00\x00\xd0>\x00\x00\xe0>\x00\x00\xf0>\x00\x00\x00?\x00\x00\x10?\x00\x00 ?\x00\x000?\x00\x00@?\x00\x00P?\x00\x00`?\x00\x00p?\x00\x00\x80?\x00\x00\x90?\x00\x00\xa0?\x00\x00\xb0?\x00\x00\xc0?\x00\x00\xd0?\x00\x00\xe0?\x00\x00\xf0?\x00\x00\x00@\x00\x00\x10@\x00\x00 @\x00\x000@\x00\x00@@\x00\x00P@\x00\x00`@\x00\x00p@\x00\x00\x80@\x00\x00\x90@\x00\x00\xa0@\x00\x00\xb0@\x00\x00\xc0@\x00\x00\xd0@\x00\x00\xe0@\x00\x00\xf0@\x00\x00\x00A\x00\x00\x10A\x00\x00 A\x00\x000A\x00\x00@A\x00\x00PA\x00\x00`A\x00\x00pA\x00\x00\x80A\x00\x00\x90A\x00\x00\xa0A\x00\x00\xb0A\x00\x00\xc0A\x00\x00\xd0A\x00\x00\xe0A\x00\x00\xf0A\x00\x00\x00B\x00\x00\x10B\x00\x00 B\x00\x000B\x00\x00@B\x00\x00PB\x00\x00`B\x00\x00pB\x00\x00\x80B\x00\x00\x90B\x00\x00\xa0B\x00\x00\xb0B\x00\x00\xc0B\x00\x00\xd0B\x00\x00\xe0B\x00\x00\xf0B\x00\x00\x00C\x00\x00\x10C\x00\x00 C\x00\x000C\x00\x00@C\x00\x00PC\x00\x00`C\x00\x00pC\x00\x00\x80C\x00\x00\x90C\x00\x00\xa0C\x00\x00\xb0C\x00\x00\xc0C\x00\x00\xd0C\x00\x00\xe0C\x00\x00\xf0C\x00\x00\x00\x80\x00\x00\x00\xbb\x00\x00\x80\xbb\x00\x00\xc0\xbb\x00\x00\x00\xbc\x00\x00 \xbc\x00\x00@\xbc\x00\x00`\xbc\x00\x00\x80\xbc\x00\x00\x90\xbc\x00\x00\xa0\xbc\x00\x00\xb0\xbc\x00\x00\xc0\xbc\x00\x00\xd0\xbc\x00\x00\xe0\xbc\x00\x00\xf0\xbc\x00\x00\x00\xbd\x00\x00\x10\xbd\x00\x00 \xbd\x00\x000\xbd\x00\x00@\xbd\x00\x00P\xbd\x00\x00`\xbd\x00\x00p\xbd\x00\x00\x80\xbd\x00\x00\x90\xbd\x00\x00\xa0\xbd\x00\x00\xb0\xbd\x00\x00\xc0\xbd\x00\x00\xd0\xbd\x00\x00\xe0\xbd\x00\x00\xf0\xbd\x00\x00\x00\xbe\x00\x00\x10\xbe\x00\x00 \xbe\x00\x000\xbe\x00\x00@\xbe\x00\x00P\xbe\x00\x00`\xbe\x00\x00p\xbe\x00\x00\x80\xbe\x00\x00\x90\xbe\x00\x00\xa0\xbe\x00\x00\xb0\xbe\x00\x00\xc0\xbe\x00\x00\xd0\xbe\x00\x00\xe0\xbe\x00\x00\xf0\xbe\x00\x00\x00\xbf\x00\x00\x10\xbf\x00\x00 \xbf\x00\x000\xbf\x00\x00@\xbf\x00\x00P\xbf\x00\x00`\xbf\x00\x00p\xbf\x00\x00\x80\xbf\x00\x00\x90\xbf\x00\x00\xa0\xbf\x00\x00\xb0\xbf\x00\x00\xc0\xbf\x00\x00\xd0\xbf\x00\x00\xe0\xbf\x00\x00\xf0\xbf\x00\x00\x00\xc0\x00\x00\x10\xc0\x00\x00 \xc0\x00\x000\xc0\x00\x00@\xc0\x00\x00P\xc0\x00\x00`\xc0\x00\x00p\xc0\x00\x00\x80\xc0\x00\x00\x90\xc0\x00\x00\xa0\xc0\x00\x00\xb0\xc0\x00\x00\xc0\xc0\x00\x00\xd0\xc0\x00\x00\xe0\xc0\x00\x00\xf0\xc0\x00\x00\x00\xc1\x00\x00\x10\xc1\x00\x00 \xc1\x00\x000\xc1\x00\x00@\xc1\x00\x00P\xc1\x00\x00`\xc1\x00\x00p\xc1\x00\x00\x80\xc1\x00\x00\x90\xc1\x00\x00\xa0\xc1\x00\x00\xb0\xc1\x00\x00\xc0\xc1\x00\x00\xd0\xc1\x00\x00\xe0\xc1\x00\x00\xf0\xc1\x00\x00\x00\xc2\x00\x00\x10\xc2\x00\x00 \xc2\x00\x000\xc2\x00\x00@\xc2\x00\x00P\xc2\x00\x00`\xc2\x00\x00p\xc2\x00\x00\x80\xc2\x00\x00\x90\xc2\x00\x00\xa0\xc2\x00\x00\xb0\xc2\x00\x00\xc0\xc2\x00\x00\xd0\xc2\x00\x00\xe0\xc2\x00\x00\xf0\xc2\x00\x00\x00\xc3\x00\x00\x10\xc3\x00\x00 \xc3\x00\x000\xc3\x00\x00@\xc3\x00\x00P\xc3\x00\x00`\xc3\x00\x00p\xc3\x00\x00\x80\xc3\x00\x00\x90\xc3\x00\x00\xa0\xc3\x00\x00\xb0\xc3\x00\x00\xc0\xc3\x00\x00\xd0\xc3\x00\x00\xe0\xc3\x00\x00\x00\x00\x00\x00\x00;\x00\x00\x80;\x00\x00\xc0;\x00\x00\x00<\x00\x00 <\x00\x00@<\x00\x00`<\x00\x00\x80<\x00\x00\x90<\x00\x00\xa0<\x00\x00\xb0<\x00\x00\xc0<\x00\x00\xd0<\x00\x00\xe0<\x00\x00\xf0<\x00\x00\x00=\x00\x00\x10=\x00\x00 =\x00\x000=\x00\x00@=\x00\x00P=\x00\x00`=\x00\x00p=\x00\x00\x80=\x00\x00\x90=\x00\x00\xa0=\x00\x00\xb0=\x00\x00\xc0=\x00\x00\xd0=\x00\x00\xe0=\x00\x00\xf0=\x00\x00\x00>\x00\x00\x10>\x00\x00 >\x00\x000>\x00\x00@>\x00\x00P>\x00\x00`>\x00\x00p>\x00\x00\x80>\x00\x00\x90>\x00\x00\xa0>\x00\x00\xb0>\x00\x00\xc0>\x00\x00\xd0>\x00\x00\xe0>\x00\x00\xf0>\x00\x00\x00?\x00\x00\x10?\x00\x00 ?\x00\x000?\x00\x00@?\x00\x00P?\x00\x00`?\x00\x00p?\x00\x00\x80?\x00\x00\x90?\x00\x00\xa0?\x00\x00\xb0?\x00\x00\xc0?\x00\x00\xd0?\x00\x00\xe0?\x00\x00\xf0?\x00\x00\x00@\x00\x00\x10@\x00\x00 @\x00\x000@\x00\x00@@\x00\x00P@\x00\x00`@\x00\x00p@\x00\x00\x80@\x00\x00\x90@\x00\x00\xa0@\x00\x00\xb0@\x00\x00\xc0@\x00\x00\xd0@\x00\x00\xe0@\x00\x00\xf0@\x00\x00\x00A\x00\x00\x10A\x00\x00 A\x00\x000A\x00\x00@A\x00\x00PA\x00\x00`A\x00\x00pA\x00\x00\x80A\x00\x00\x90A\x00\x00\xa0A\x00\x00\xb0A\x00\x00\xc0A\x00\x00\xd0A\x00\x00\xe0A\x00\x00\xf0A\x00\x00\x00B\x00\x00\x10B\x00\x00 B\x00\x000B\x00\x00@B\x00\x00PB\x00\x00`B\x00\x00pB\x00\x00\x80B\x00\x00\x90B\x00\x00\xa0B\x00\x00\xb0B\x00\x00\xc0B\x00\x00\xd0B\x00\x00\xe0B\x00\x00\xf0B\x00\x00\x00C\x00\x00\x10C\x00\x00 C\x00\x000C\x00\x00@C\x00\x00PC\x00\x00`C\x00\x00pC\x00\x00\x80C\x00\x00\x90C\x00\x00\xa0C\x00\x00\xb0C\x00\x00\xc0C\x00\x00\xd0C\x00\x00\xe0C\x00\x00\xf0C\x00\x00\x00\x80\x00\x00\x00\xbb\x00\x00\x80\xbb\x00\x00\xc0\xbb\x00\x00\x00\xbc\x00\x00 \xbc\x00\x00@\xbc\x00\x00`\xbc\x00\x00\x80\xbc\x00\x00\x90\xbc\x00\x00\xa0\xbc\x00\x00\xb0\xbc\x00\x00\xc0\xbc\x00\x00\xd0\xbc\x00\x00\xe0\xbc\x00\x00\xf0\xbc\x00\x00\x00\xbd\x00\x00\x10\xbd\x00\x00 \xbd\x00\x000\xbd\x00\x00@\xbd\x00\x00P\xbd\x00\x00`\xbd\x00\x00p\xbd\x00\x00\x80\xbd\x00\x00\x90\xbd\x00\x00\xa0\xbd\x00\x00\xb0\xbd\x00\x00\xc0\xbd\x00\x00\xd0\xbd\x00\x00\xe0\xbd\x00\x00\xf0\xbd\x00\x00\x00\xbe\x00\x00\x10\xbe\x00\x00 \xbe\x00\x000\xbe\x00\x00@\xbe\x00\x00P\xbe\x00\x00`\xbe\x00\x00p\xbe\x00\x00\x80\xbe\x00\x00\x90\xbe\x00\x00\xa0\xbe\x00\x00\xb0\xbe\x00\x00\xc0\xbe\x00\x00\xd0\xbe\x00\x00\xe0\xbe\x00\x00\xf0\xbe\x00\x00\x00\xbf\x00\x00\x10\xbf\x00\x00 \xbf\x00\x000\xbf\x00\x00@\xbf\x00\x00P\xbf\x00\x00`\xbf\x00\x00p\xbf\x00\x00\x80\xbf\x00\x00\x90\xbf\x00\x00\xa0\xbf\x00\x00\xb0\xbf\x00\x00\xc0\xbf\x00\x00\xd0\xbf\x00\x00\xe0\xbf\x00\x00\xf0\xbf\x00\x00\x00\xc0\x00\x00\x10\xc0\x00\x00 \xc0\x00\x000\xc0\x00\x00@\xc0\x00\x00P\xc0\x00\x00`\xc0\x00\x00p\xc0\x00\x00\x80\xc0\x00\x00\x90\xc0\x00\x00\xa0\xc0\x00\x00\xb0\xc0\x00\x00\xc0\xc0\x00\x00\xd0\xc0\x00\x00\xe0\xc0\x00\x00\xf0\xc0\x00\x00\x00\xc1\x00\x00\x10\xc1\x00\x00 \xc1\x00\x000\xc1\x00\x00@\xc1\x00\x00P\xc1\x00\x00`\xc1\x00\x00p\xc1\x00\x00\x80\xc1\x00\x00\x90\xc1\x00\x00\xa0\xc1\x00\x00\xb0\xc1\x00\x00\xc0\xc1\x00\x00\xd0\xc1\x00\x00\xe0\xc1\x00\x00\xf0\xc1\x00\x00\x00\xc2\x00\x00\x10\xc2\x00\x00 \xc2\x00\x000\xc2\x00\x00@\xc2\x00\x00P\xc2\x00\x00`\xc2\x00\x00p\xc2\x00\x00\x80\xc2\x00\x00\x90\xc2\x00\x00\xa0\xc2\x00\x00\xb0\xc2\x00\x00\xc0\xc2\x00\x00\xd0\xc2\x00\x00\xe0\xc2\x00\x00\xf0\xc2\x00\x00\x00\xc3\x00\x00\x10\xc3\x00\x00 \xc3\x00\x000\xc3\x00\x00@\xc3\x00\x00P\xc3\x00\x00`\xc3\x00\x00p\xc3\x00\x00\x80\xc3\x00\x00\x90\xc3\x00\x00\xa0\xc3\x00\x00\xb0\xc3\x00\x00\xc0\xc3\x00\x00\xd0\xc3\x00\x00\xe0\xc3', dtype=np.float32)    
    arr2 = np.empty(arr.shape[0], dtype="float32")
    for i in prange(arr.shape[0]):
            arr2[i] = fp8table[arr[i]]
    return arr2    


def tofp32(arr):
    """Converts a fp8 (4M3E) array to fp32.
    Reshapes the array to be one
    dimensional and uses a numba-optimized function
    """
    return tofp32n8(arr.reshape(arr.shape[0]*arr.shape[1])).reshape(arr.shape)


@njit('uint8[:](uint32[:])', parallel=True)
def tofp8n(arr):
    """Numba-optimized function that converts an array of fp32 to fp8 (4M3E) 
    Uses the algorithm described by ProjectPhysX at https://stackoverflow.com/questions/1659440/32-bit-to-16-bit-floating-point-conversion
    and https://www.researchgate.net/publication/362275548_Accuracy_and_performance_of_the_lattice_Boltzmann_method_with_64-bit_32-bit_and_customized_16-bit_number_formats
    """
    arr2 = np.empty(arr.shape[0], dtype="uint8")
    for i in prange(arr.shape[0]):
            # round-to-nearest-even: add last bit after truncated mantissa (1+8+3) from left
            y = arr[i] + 0x00080000 
            e = (y&0x7F800000)>>23 # exponent
            m =  y&0x007FFFFF #mantissa
            
            if e > 135:
                arr2[i] = 0x7F | (y&0x80000000)>>24 # saturated
            elif e > 120:
                arr2[i] = ((e-120)<<3) & 0x78 | m>>20 | (y&0x80000000)>>24 # normalized
            elif e < 121 and e > 116:
                # 0x00780000 = 0x00800000-0x00080000 = decimal indicator flag - initial rounding
                arr2[i] = ((((m+0x00780000)>>(140-e))+1)>>1) | (y&0x80000000)>>24
            else:
                arr2[i] = 0 | (y&0x80000000)>>24
    return arr2 


def tofp8(arr):
    """Converts an array of fp32 to fp8 (4M3E)
    Reshapes the array to be one
    dimensional and uses a numba-optimized function   
    """
    return tofp8n(arr.view(dtype=np.uint32).reshape(arr.shape[0]*arr.shape[1])).view(dtype=np.uint8).reshape(arr.shape)



class BlobTable():
    """Class to handle a storage of variable-length values of a key-value storage
    Key is fixed length of key_length
    """
    def __init__(self, store, key_length=20):
        """Initializes class using a pytables store and a key_length value
        """
        if "keys" not in store.root:
            # reasonable compression optimized for reading speed
            filters = tables.Filters(complevel=5, complib='blosc:lz4', 
                                 shuffle=1, bitshuffle=0)
        
            blob_type = {"key": tables.StringCol(key_length, pos=0),
                         "offset":tables.Int64Col(pos=1), 
                         "length": tables.Int64Col(pos=2),
                         }

            self.keys_table = store.create_table("/", "keys", 
                                blob_type, 
                                filters=filters, 
                                chunkshape=10000)

            # high compression optimized with a reading speed compromise
            filters = tables.Filters(complevel=5, complib='blosc:zstd', 
                         shuffle=1, bitshuffle=0)            
            self.values_table = store.create_earray("/", "values", atom=tables.UInt8Atom(), shape=(0,), filters=filters)  
        else:
            self.keys_table = store.root.keys
            self.values_table = store.root.values
            
        self.offset = self.values_table.nrows
        self.nrows = self.keys_table.nrows
        self._is_closed = False
    
    
    def __len__(self):
        return self.nrows
    
    def create_index(self):
        self.keys_table.cols.key.reindex()
    
    
    def append(self, key, value):
        """Appends a key-value to the storage
        """
        # store variable length value
        length = len(value)
        self.values_table.append(np.frombuffer(value, dtype=np.uint8))

        # store index
        row = self.keys_table.row
        row["key"] = key
        row["offset"] = self.offset
        row["length"] = length
        row.append()
        self.offset += length
        self.nrows += 1

    def __getitem__ (self, rownum):
        if isinstance(rownum, slice):
            return [self[ii] for ii in range(*rownum.indices(len(self)))]
        else:
            row = self.keys_table[rownum]
            offset = row['offset']
            value = self.values_table.read(offset, offset+row["length"]).tobytes()
        
        return value


    def get_value (self, key):
        key = key.encode("utf8")
        offset, length = [(r['offset'], r['length']) for r in self.keys_table.where(f"key=={key}")][0]
        value = self.values_table.read(offset, offset+length).tobytes()
        return value


class Simiandb():
    """Wrapper around pytables store .
    To use, you should have the ``pytables`` python package installed.
    Example:
        .. code-block:: python
                from simiandb import Simiandb
                docdb = Simiandb("store")
    """

    def __init__(self, storepath, embedding_function=None,  mode="a", id_length = 19):
        
        if mode not in ["a", "w", "r"]:
            raise ValueError("Mode can only be r, w or a")
        self._embedding_function = embedding_function
        self._storename = Path(storepath)
        self._mode = mode
        if not self._storename.exists():
            self._storename.mkdir()
        
        self._vectorstore = tables.open_file( self._storename / "embeddings.h5", mode = mode)
        self._docstore = tables.open_file( self._storename / "documents.h5", mode = mode)
        self._metastore = tables.open_file( self._storename / "metadatas.h5", mode = mode)
        self._embedding_function = embedding_function
        self._is_closed = False
        if 'embeddings' in self._vectorstore.root:
            self._vector_table = self._vectorstore.root.embeddings
        self._docs_table = BlobTable(self._docstore, id_length)
        return
    
    
    def __enter__(self):
        """Magic method Required for usage with the with statement
        """
        return self
        

    def _get_top_indexes(self, c, k):
        count = self._vector_table.nrows
        st =0 
        batch = self._vector_table.chunkshape[0]*25
        res = np.ascontiguousarray(np.empty(shape=(count,), dtype="float32"))
        end = 0
        # a = time()
        while end!=count:
            end += batch
            end = end if end <= count else count 
            t_res = structured_to_unstructured(self._vector_table.read(start=st, stop=end))
            t_res = tofp32(t_res)
            np.dot(t_res,c, res[st:end])
            st = end

        indices = np.argpartition(res, -k)[-k:] #from https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        indices = indices[np.argsort(res[indices])[::-1]]
        # print(time() -a)
        return indices


    def _create_embeddings_table(self, dimensions):
        """Creates the embeddings table within the pytables file
        """
        if dimensions > 512:
            # prevent pytables warning on max_columns
            tables.parameters.MAX_COLUMNS = len(dimensions)
        embedding_type = {f"d{n}":tables.UInt8Col(pos=n) for n in range(dimensions)}
        
        # no compression for embeddings
        filters = None
        
        self._vector_table = self._vectorstore.create_table("/", "embeddings", 
                                           embedding_type, 
                                           filters=filters, 
                                           chunkshape=10000)


    def _check_closed(self):
        if self._is_closed:
            raise ValueError("Simiandb is already closed")


    def add_texts(self, texts, metadatas = None, ids = None, embeddings=None, show_progress_bar=True):
        """Run more texts through the embeddings and add to the vectorstore.
        Args:
            texts (Iterable[str]): Texts to add to the vectorstore.
            metadatas (Optional[List[dict]], optional): Optional list of metadatas.
            ids (Optional[List[str]], optional): Optional list of IDs.
            embeddings (Optional[List[array]], optional): Optional list of embeddings.
        Returns:
            List[str]: List of IDs of the added texts.
        """

        self._check_closed()
      
        self._add_embeddings(texts, embeddings, show_progress_bar)
        
        if ids is None:
            ids = list(range(self._docs_table.nrows, self._docs_table.nrows + len(texts)))
            
        for textid, text in zip(ids, texts):
            self._docs_table.append(textid, text.encode("utf8"))

        return ids


    def get_text(self, key):
        return self._docs_table.get_value(key).decode("utf8")
    
    
    def create_keys_index(self):
        self._docs_table.create_index()
    

    def _add_embeddings(self, texts, embeddings, show_progres_bar):
        """Calculate or use embeddings to fill the embeddings table
        """
        
        if embeddings is None and self._embedding_function is not None:
            embeddings = self._embedding_function.embed_documents(texts, show_progres_bar=show_progres_bar)
        
        if embeddings is not None and 'embeddings' not in self._vectorstore.root:
            dimensions = len(embeddings[0])
            self._create_embeddings_table(dimensions)
        
        if  embeddings is not None :
            self._vector_table = self._vectorstore.root.embeddings
               
            embeddings = tofp8(np.array(embeddings, dtype=np.float32))
            self._vector_table.append(embeddings)


    def regenerate_embeddings(self, embeddings=None, show_progress_bar=True):
        """Run existing texts through the embeddings and add to the vectorstore.
        Args:
            embeddings (Optional[List[array]], optional): Optional list of embeddings.
        """

        self._check_closed()
        self._vectorstore.close()
        (self._storename / "embeddings.h5").kill()
        self._vectorstore = tables.open_file( self._storename / "embeddings.h5", mode = self._mode)
        
        batch_size = 1000
        for i in tqdm(range(0, len(self._docs_table), batch_size), disable=not show_progress_bar):
            text_batch = [text.decode("utf8") for text in self._docs_table[i:i+batch_size]]
            if embeddings is not None:
                embeddings_batch = embeddings[i:i+batch_size]
            elif self.embedding_function is not None:
                embeddings_batch = self._embedding_function.embed_documents(text_batch)
            else:
                raise ValueError("Neither embeddings nor embedding function provided")
            self._add_embeddings(text_batch, embeddings_batch, show_progress_bar)
        return 


    def similarity_search(self, query: str, k = 4, filter = None):
        """Run similarity search with PytableStore.
        Args:
            query (str): Query text to search for.
            k (int): Number of results to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
        Returns:
            List[Document]: List of documents most simmilar to the query text.
        """
        self._check_closed()
        query_embedding = np.array(self._embedding_function.embed_query(query),dtype="float32")
        results = self._get_top_indexes(query_embedding, k)

        docs = [self._docs_table[i].decode("utf8") for i in results]
        return docs


    def close(self):
        """Makes sure the pytables file is closed
        """
        if not self._is_closed:
            self._is_closed = True
            
            if hasattr(self, '_vectorstore'):
                try:
                    self._vectorstore.flush()
                    self._docstore.flush()
                    self._metastore.flush()
                    self._vectorstore.close()
                    self._docstore.close()
                    self._metastore.close()
                except:
                    print("Unable to close file")


    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Magic method Required for usage with the with statement
        """
        self.close()

    def __del__(self):
        """Magic method just in case the object is deleted without closing it
        """
        self.close()
  


if __name__ == '__main__':
    pass
    with Simiandb("mystore",mode="w") as docdb:
        pass