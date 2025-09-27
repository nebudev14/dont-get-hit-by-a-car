from fastapi import FastAPI, Request, Response, status

app = FastAPI()

@app.middleware("http")
async def checkphone(request: Request, call_next):
    platform = request.headers.get("sec-ch-ua-platform")   # e.g., "Android"
    model    = request.headers.get("sec-ch-ua-model")      # e.g., "Pixel 8"

    is_pixel = (
        platform and "android" in platform.lower()
        and model and "pixel" in model.lower()
    )

    if not is_pixel:
        # Stop the request here
        return Response(status_code=status.HTTP_403_FORBIDDEN)

    # Optional: stash for endpoints/dependencies
    request.state.is_pixel = True

    # Continue to next middleware/route
    response = await call_next(request)
    return response


# @app.post("/camera/stream", status_code=status.HTTP_200_OK)
# async def camera_stream(request: Request):
    
    



