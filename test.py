import inspect

class A:
    @staticmethod
    def func():
        frame = inspect.currentframe()
        print(frame)
        infer_scope_caller = frame.f_back
        print(infer_scope_caller)
        filename = inspect.getmodule(infer_scope_caller).__name__
        print(filename)
        split_filename = filename.split('.')
        return  split_filename[0]

if __name__ == '__main__':
    print(A.func())