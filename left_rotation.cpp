#include <bits/stdc++.h>
using namespace std;
vector<int> rotateLeft(int d, vector<int> a) {
    int len=a.size();
     while(d--){
         int z=a[0];
        for(int i=0;i<len-1;i++){
            a[i]=a[i+1];
        }
        a[len-1]=z;
     }
     return a;
}
int main(){
    int n,k; cin>>n>>k;
   vector<int>a(n);
   for(int i=0;i<n;i++){
       cin>>a[i];
   }
    vector<int>b = rotateLeft(k,a);
    for(int x:b){
        cout<<x<<" ";
    }
}
