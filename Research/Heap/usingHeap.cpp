#include <bits/stdc++.h>
using namespace std;
int main()
{
    vector<int>v1={10,20,21,22,232,1};
    make_heap(v1.begin(),v1.end());
    cout<<v1.front()<<endl;
    return 0;
}
