import java.util.concurrent.Semaphore;

class MST{

    Semaphore s1 = new Semaphore(1);
    Semaphore s2 = new Semaphore(0);
    public void foo(){
        try{
            s1.acquire();
        }
        catch(InterruptedException ie){}
        
        System.out.println("foo");
        s2.release();
    }

    public void bar(){
        try{
            s2.acquire();
        }
        catch(InterruptedException ie){}
        System.out.println("bar");
        s1.release();
    }

    public static void main(String[] args){
        MST ob = new MST();
        Thread t1 = new Thread(() -> {
            for(int i = 0; i < 10; i++){
                ob.foo();
            }
        });

        Thread t2 = new Thread(() -> {
            for(int i = 0; i < 10; i++){
                ob.bar();
            }
        });

        t1.start();
        t2.start();

        try{
            t1.join();
            t2.join();
        }    
        catch(InterruptedException ie){}    
    }
}