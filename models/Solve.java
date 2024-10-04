import java.util.Scanner;

class Solve{

    public static void main(String args[]){
        Scanner ip = new Scanner(System.in);
        int ans = 0;

        while(ip.hasNextLine()){
            String s = ip.nextLine();
            StringBuilder sb = new StringBuilder(s);
            int num = 0;
            num += helper(s);
            num *= 10;
            num += helper(sb.reverse().toString());
            
            ans += num;
        }
        System.out.println(ans);
    }

    public static int helper(String s){
        
    }
}