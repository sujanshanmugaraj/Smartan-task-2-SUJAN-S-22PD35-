# yolov5/utils/common_utils.py

def TryExcept(func):
    try:
        return func()
    except Exception as e:
        print(f"Error: {e}")
        return None

def threaded(func):
    from threading import Thread
    def wrapper(*args, **kwargs):
        thread = Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
        return thread
    return wrapper
