import java.util.*;
import java.util.HashMap;

class Try{
    public static void main(String[] args){
        Graph g = new Graph();
        g.addEdge(new int[]{1,2});
        g.addEdge(new int[]{1,3});
        g.addEdge(new int[]{1,4});
        g.addEdge(new int[]{3,5});
        g.addEdge(new int[]{5,3});
        g.printGraph();
    }
}

class Graph{
    HashMap<Integer, List<Integer>> graph;

    public Graph(){
        graph = new HashMap<>();
    }

    public void addEdge(int[] edge){
        int src = edge[0];
        int dsnt = edge[1];

        graph.putIfAbsent(src, new ArrayList<>());
        graph.get(src).add(dsnt);
    }

    void printGraph(){
        for(int n : graph.keySet()){
            List<Integer> adjL = graph.get(n);
            System.out.print(n + ": ");
            for(int i = 0; i < adjL.size(); i++){
                System.out.print(adjL.get(i) + " ");
            }
            System.out.println();
        }
    }
}