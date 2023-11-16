import time
import http.client
import json
import numpy.random as npr

_last_call = 0
REQ_PER_SEC = 5
_erc721_cached = {}

with open("api_key.txt") as f:
    key = f.readline().strip()

url = "api.etherscan.io"
connection = http.client.HTTPSConnection(url)


def rate_limit(limit):
    """
    Decorator function uses global variable to make sure all functions are called < rate limit/second (enforced by EtherScan)
    This function is blocking: will cause code to sleep until it's valid to call function

    """

    def inner(func):
        def wrapper(*args, **kwargs):
            global _last_call
            curr_time = time.time()
            # if curr_time - last_call > 1/limit:
            #     func(*args, **kwargs)
            #     last_call = time.time()
            # else:
            #     return None
            if curr_time - _last_call < 1 / limit:
                time.sleep(1 / limit - curr_time + _last_call)
            res = func(*args, **kwargs)
            _last_call = time.time()
            return res

        return wrapper

    return inner


# maximize offset=10000 bc of rate limit of api calls (5/second)
# 3 modes atleast one of user_adr,contr_adr has to be present
@rate_limit(REQ_PER_SEC)
def get_ERC721_trans(user_adr=None, contr_addr=None, offset=10000, debug=False):
    """
    Call EtherScan API to get ERC721 transactions
    Return: entire transaction dictionary
    """
    if user_adr is None and contr_addr is None:
        return None

    api = f"/api\
   ?module=account\
   &action=tokennfttx"
    if contr_addr:
        api += f"\
   &contractaddress={contr_addr}"
    if user_adr:
        api += f"\
   &address={user_adr}"

    api += f"\
   &page=1\
   &offset={offset}\
   &startblock=0\
   &endblock=27025780\
   &sort=asc\
   &apikey={key}"

    api = api.replace(" ", "")
    if debug:
        print(api)
    connection.request("GET", api, body=None)
    response = connection.getresponse()
    response = response.read().decode()
    # if debug:
    # print(response[:400])
    response = json.loads(response)
    if debug:
        # print(response)
        print("RESULT: ", response["result"])
    return response["result"]


# space inefficient, but far more time efficient
# don't have error handling but its ok:
# if we make a call to none, none we will store in cache as None and don't need to make another call
# at worst losing 0.2 sec
def get_ERC721_tx_cached(user_adr=None, contr_addr=None, offset=10000, debug=False):
    global _erc721_cached
    if (user_adr, contr_addr) in _erc721_cached:
        return _erc721_cached[(user_adr, contr_addr)]
    else:
        _erc721_cached[(user_adr, contr_addr)] = get_ERC721_trans(
            user_adr, contr_addr, offset=offset
        )
        if debug:
            print("CACHED: ", _erc721_cached)
        return _erc721_cached[(user_adr, contr_addr)]


# next step on directed graph: find random address that has received an ERC721 transfer
def get_rand_neighbour(user_adr, contr_addr=None, use_cached=True):
    return npr.choice(get_neighbours(user_adr, contr_addr, use_cached))


def get_neighbours(user_adr, contr_addr=None, use_cached=True, offset=10000):
    if use_cached:
        tx_list = get_ERC721_tx_cached(user_adr, contr_addr, offset=offset)
    else:
        tx_list = get_ERC721_trans(user_adr, contr_addr, offset=offset)

    recipient_addrs = [tx["to"] for tx in tx_list] if tx_list is not None else []
    return recipient_addrs


# get list of all users that have a token minted
def get_tx_contract(contr_addr, use_cached=True, offset=10000, debug=False):
    if use_cached:
        tx_list = get_ERC721_tx_cached(
            user_adr=None, contr_addr=contr_addr, offset=offset
        )
    else:
        tx_list = get_ERC721_trans(user_adr=None, contr_addr=contr_addr, offset=offset)
    if debug:
        print("TX LEN: ", len(tx_list))
    recipient_addrs = [tx["to"] for tx in tx_list] if tx_list is not None else []
    return recipient_addrs
