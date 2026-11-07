"""Real loopback HTTP/SSE tests with a deterministic CPU-only fake core."""
import http.client,json,pathlib,sys,threading,types,unittest
from unittest.mock import patch
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from test_async_engine import FakeCore
from engine.async_engine import AsyncEngine
from server import Handler,ThreadingHTTPServer
class Tokenizer:
    def encode(self,*a,**k):return [1,2]
    def apply_chat_template(self,*a,**k):return 'prompt'
    def decode(self,ids,**kw):
        return '' if not ids else ('\ufffd' if len(ids)==1 else '中'+('x'*(len(ids)-2)))
class HTTPTests(unittest.TestCase):
    def setUp(self):
        tok=Tokenizer();FakeCore.gate=None;FakeCore.fail=False
        self.patch=patch.dict(sys.modules,{'transformers':types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a,**k:tok)),
            'engine.core':types.SimpleNamespace(EngineCore=FakeCore)})
        self.patch.start();self.engine=AsyncEngine('fake');self.engine.start()
        Handler.engine=self.engine;Handler.tokenizer=tok;Handler.metrics_path=None
        Handler.processor=None
        self.srv=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.t=threading.Thread(target=self.srv.serve_forever);self.t.start()
    def tearDown(self):
        self.srv.shutdown();self.srv.server_close();self.t.join();self.engine.stop();self.patch.stop()
    def request(self,body):
        c=http.client.HTTPConnection('127.0.0.1',self.srv.server_port,timeout=3)
        c.request('POST','/v1/chat/completions',json.dumps(body),{'Content-Type':'application/json'})
        r=c.getresponse();data=r.read();code=r.status;c.close();return code,data
    def test_stream_unicode_done_and_nonstream(self):
        body=dict(messages=[dict(role='user',content='hello')],max_tokens=3,temperature=0)
        code,raw=self.request(dict(body,stream=True));self.assertEqual(code,200)
        lines=[s[6:] for s in raw.decode().splitlines() if s.startswith('data: ')]
        self.assertEqual(lines[-1],'[DONE]')
        chunks=[json.loads(s) for s in lines[:-1]]
        text=''.join(c['choices'][0]['delta'].get('content','') for c in chunks)
        self.assertEqual(text,'中x');self.assertEqual(chunks[-1]['choices'][0]['finish_reason'],'length')
        code,raw=self.request(body);self.assertEqual(code,200)
        obj=json.loads(raw);self.assertEqual(obj['choices'][0]['message']['content'],text)
        self.assertEqual(obj['usage']['completion_tokens'],3)
        self.assertEqual(self.engine.stats()['active'],0)
    def test_validation_and_worker_error(self):
        body=dict(messages=[dict(role='user',content='hello')],max_tokens=3,temperature=0)
        for bad in (dict(body,max_tokens=0),dict(body,temperature='nan'),dict(body,top_p=.8),dict(body,messages=[dict(role='user',content=[])])):
            self.assertEqual(self.request(bad)[0],400)
        FakeCore.fail=True
        code,raw=self.request(body);self.assertEqual(code,500)
        self.assertIn('injected worker',raw.decode())
    def test_inline_image_passes_through_http_and_engine(self):
        import base64,io
        from PIL import Image
        buf=io.BytesIO();Image.new('RGB',(64,96),'red').save(buf,format='PNG')
        class ImageTokenizer(Tokenizer):
            unk_token_id=-1
            def convert_tokens_to_ids(self,s):return 7
        Handler.processor=types.SimpleNamespace(tokenizer=ImageTokenizer(),
            apply_chat_template=lambda *a,**kw:'prefix<|image_pad|>suffix')
        media=dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()))
        body=dict(messages=[dict(role='user',content=[media,dict(type='text',text='what color?')])],
                  max_tokens=3,temperature=0)
        code,raw=self.request(body)
        self.assertEqual(code,200,raw)
        mm=self.engine.engine.media_received[0]
        self.assertEqual(mm['images'][0]['grid'],[1,6,4])
        self.assertEqual(mm['images'][0]['pixels'].shape,(24,1536))
        self.assertEqual(mm['pos3'].shape,(3,10))
        self.assertEqual(json.loads(raw)['usage']['prompt_tokens'],10)
if __name__=='__main__':unittest.main()
