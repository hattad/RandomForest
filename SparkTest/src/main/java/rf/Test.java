package rf;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Set;


public class Test {
 public static void main(String args[]){
	 /*int numFeatures = 10;
	 Set<Integer> featureIndex = new HashSet<Integer>();
	 if(0 != numFeatures){
			for(int i=0; i < numFeatures; i++){
				//if(featureIndex.size()<10) {
					featureIndex.add((int) (Math.random()*numFeatures));
				//}
			}
		}
	 for(int i : featureIndex){
		 System.out.println(i);
	 }*/
	/* 
	    List<Integer> dataList = new ArrayList<Integer>();
	    for (int i = 0; i < 10; i++) {
	      dataList.add(i);
	    }
	    Collections.shuffle(dataList);
	    dataList.toArray();
	    Integer[] num = (Integer[])dataList.toArray();
	    for (int i = 0; i < dataList.size(); i++) {
	      num[i] = dataList.get(i);
	    }

	    for (int i = 0; i < num.length; i++) {
	      System.out.println(num[i]);
	    }*/
	 
/*	 List<Integer> data = new ArrayList<Integer>();
	 data.add(1);
	 data.add(2);
	 data.add(4);
	 data.add(3);
	 for(Integer num : data){
		 System.out.println(num);
	 }
	 Iterator<Integer> it = data.iterator();
	 while(it.hasNext()){
		 System.out.println(it.next());
	 }*/
	/* Random random = new Random(); 
	 //random.setSeed(System.currentTimeMillis());
	 
	 final Set<Integer> featureIndex = new HashSet<Integer>();
		//Generate random numbers of size equal to the number of trees * number of attributes
		//Integer featureSize = 3 * 10;
			//for(int i=0; i < 10; i++){
				while(featureIndex.size() < 10) {
					random.setSeed(System.currentTimeMillis());
					//System.out.println(random.nextInt(10));
					//int num = (int) (Math.random()*(10));
					featureIndex.add(random.nextInt(10));
					//System.out.println(num);
			   }
			//}
		for(int num : featureIndex){
			System.out.println(num);
		}*/
	 while(true){
	 switch((int) (Math.random()*10)) {
		case 0: System.out.println(0);
				break;
		case 1: System.out.println(1);
				break;
		case 2: System.out.println(2);
				break;
		case 3: System.out.println(3);
				break;
}
	 }
 }
}
