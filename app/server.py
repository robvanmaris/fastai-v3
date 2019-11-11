import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.text import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://www.dropbox.com/s/6bgq8t6yextloqp/export.pkl?raw=1'
export_file_url2 = 'https://www.googleapis.com/drive/v3/files/1-EspcL2g7QlYBlrevM_oPA8LWsVWl2kj?alt=media&key=AIzaSyCiXhcX53r8hEv1qfQmKcWsxRPp18Pey5c'
export_file_name = 'export.pkl'

classes = ['black', 'grizzly', 'teddys']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

REGEX_TEXT_BETWEEN_DOUBLE_QUOTES = re.compile(r'"\s(.*?)\s"')
async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url2, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/generate')
async def generate(request):
    start = request_param(request, 'start', 'xxbos')
    words = int(request_param(request, 'words', 500))
    temp = request_param(request, 'temp', 0.75)

    prediction = learn.predict(start, words, temperature=temp)
    prediction = REGEX_TEXT_BETWEEN_DOUBLE_QUOTES.sub(r'"\1"', prediction)
    prediction = (prediction.replace(" .", ".")
                        .replace(" ,", ",")
                        .replace(" )", ")")
                        .replace("( ", "("))
    return JSONResponse({'result': str(prediction)})
 
def request_param(request, name, default):
    return request.query_params.get(name, default)

def link(s):
    sanitized = s.replace(' ', '+')
    return f'/generate?start={sanitized}'

async def predict(learner, text:str, n_words:int=1, no_unk:bool=True, temperature:float=1., min_p:float=None, sep:str= ' ',
            decoder=decode_spec_tokens):
    "Return the `n_words` that come after `text` as a stream of words separated by linebreaks"
    learner.model.reset()
    xb,yb = learner.data.one_item(text)
    new_idx = []
    for i in range(n_words):
        res = learner.pred_batch(batch=(xb, yb))[0][-1]
        if no_unk: res[learner.data.vocab.stoi[UNK]] = 0.
        if min_p is not None:
            if (res >= min_p).float().sum() == 0:
                warn(f"There is no item with probability >= {min_p}, try a lower value.")
            else: res[res < min_p] = 0.
        if temperature != 1.: res.pow_(1 / temperature)
        idx = torch.multinomial(res, 1).item()
        new_idx.append(idx)
        xb = xb.new_tensor([idx])[None]
        next_word = learner.data.vocab.textify([idx], sep=None)[0]
        if not next_word in [TK_MAJ, TK_UP, TK_REP, TK_WREP]:
            yield decoder(learner.data.vocab.textify(new_idx, sep=None))[0] + '\n'
            new_idx = []
            await asyncio.sleep(0)

@app.route('/test')
async def test(request):
    start = request_param(request, 'start', 'xxbos')
    words = int(request_param(request, 'words', 500))
    temp = request_param(request, 'temp', 0.75)

    prediction = predict(learn, start, words, temperature=temp)
    return StreamingResponse(prediction, media_type='text/plain')

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
