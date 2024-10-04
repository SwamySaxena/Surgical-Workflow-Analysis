#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *reverseSentence(char *Str){
    int i, len, temp;  
    len = strlen(Str);
      
    for (i = 0; i < len/2; i++)  
    {  
        temp = Str[i];  
        Str[i] = Str[len - i - 1];  
        Str[len - i - 1] = temp;  
    }  

    char *ret = Str;
    return ret;
}

int main(int argc, char *argv[]){
    char *Str = argv[1];
    printf("%s\n", reverseSentence(Str));
    return 0;
}