import asyncio
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))


def test_vlm_omitted_defaults_and_explicit_zero_survive_chunks():
    from mlx_vlm_server import PolicyMiddleware
    seen = []
    async def app(scope, receive, send):
        seen.append(json.loads((await receive())['body']))
    async def run():
        chunks = iter([{'type':'http.request','body':b'{"temperature":0,', 'more_body':True}, {'type':'http.request','body':b'"presence_penalty":0,"messages":[]}', 'more_body':False}])
        async def receive(): return next(chunks)
        async def send(message): pass
        await PolicyMiddleware(app, {'sampling':{'temperature':1.0,'top_k':64,'presence_penalty':1.5}})({'type':'http','method':'POST','path':'/v1/chat/completions','headers':[]}, receive, send)
    asyncio.run(run())
    assert seen == [{'temperature':0,'presence_penalty':0,'messages':[],'top_k':64}]


def test_vlm_invalid_json_preserved():
    from mlx_vlm_server import PolicyMiddleware
    seen=[]
    async def app(scope,receive,send): seen.append((await receive())['body'])
    async def receive(): return {'type':'http.request','body':b'invalid','more_body':False}
    async def send(message): pass
    asyncio.run(PolicyMiddleware(app,{'sampling':{'temperature':1.0}})({'type':'http','method':'POST','path':'/v1/chat/completions','headers':[]},receive,send))
    assert seen == [b'invalid']


def test_nested_thinking_override_selects_recipe_and_vlm_wire_flag():
    from mlx_vlm_server import PolicyMiddleware
    seen=[]
    async def app(scope,receive,send): seen.append(json.loads((await receive())['body']))
    async def receive(): return {'type':'http.request','body':b'{"chat_template_kwargs":{"enable_thinking":true}}','more_body':False}
    async def send(message): pass
    policy={'sampling':{'enable_thinking':False,'temperature':.7},'sampling_recipes':{'thinking':{'sampling':{'temperature':1.}}}}
    asyncio.run(PolicyMiddleware(app,policy)({'type':'http','method':'POST','path':'/v1/chat/completions','headers':[]},receive,send))
    assert seen[0]['enable_thinking'] is True
    assert seen[0]['temperature'] == 1.
