#include <iostream>
#include <vector>

using v = std::vector<int>;

#define L(k) ((k?k:new v()) -> size())

int foo(v* s = NULL){
    static v* x = NULL;
    v* y = NULL;
    static int r;

    s?x=s, y=s:s;

    r += L(x);
    r -= L(y);

    return r;
}

int main(){
    v vec1 = {1,2,3,4,5};
    v vec2 = {4,5,6};
    foo(&vec1); foo();
    foo(&vec2); foo();

    std::cout << "answer: " << foo() << std::endl;
    return 0;
}