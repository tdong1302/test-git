#include <bits/stdc++.h>
using namespace std;
vector<int> reverseArray(vector<int> a) {
    int n=a.size();
    for(int i=0;i<n/2;i++){
        int z=a[i];
        a[i]=a[n-i-1];
        a[n-i-1]=z;
    }
    return a;
}
int main(){
    vector<int> a;
    int n,tmp;
    cin>>n;
    for(int i=0;i<n;i++){
        cin>>tmp;
        a.push_back(tmp);
    }
    vector<int>b=reverseArray(a);
    int len=b.size();
    for(int i=0;i<len;i++){
        cout<<b[i]<<" ";    
    }
}