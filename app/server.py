import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
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

if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
