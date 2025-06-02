import math
from mcp.server.fastmcp import FastMCP
from typing import List

# initialize server
mcp=FastMCP("math_server")

@mcp.tool()
def bezout(a: int,b: int) -> List[int]:
    """ Compute the Bezout coefficients of a and b
        Args:
            a, b: input integers
        Returns:
            s,t with as+bt=gcd(a,b)
    """
    # swap a and b if necessary
    swap=False # indicate if we need to swap and b
    if a<b:
        swap=True
        a,b=b,a

    # keep track the list of remainders, coefficients a[i] and b[i], quotients q[i]
    remainders=[a,b]   # store remainders r[0], r[1], r[2], ...
    coeff_a=[1,0]      # store coefficients a[0], a[1], a[2], ...
    coeff_b=[0,1]      # store coefficients b[0], b[1], b[2], ...
    quotients=list()   # store quotients q[0], q[1], q[2], ...

    while b>0:
        # continously divide a by b and update them
        q=a//b
        a,b=b,a-b*q

        # update the lists
        remainders.append(b)
        quotients.append(q)
        coeff_a.append(coeff_a[-2]-q*coeff_a[-1])
        coeff_b.append(coeff_b[-2]-q*coeff_b[-1])

    if swap:
        return [coeff_b[-2], coeff_a[-2]]
    else:
        return [coeff_a[-2], coeff_b[-2]]


@mcp.tool()
def solve(a: int,b: int,m: int) -> str:
    """ Solve the linear congruence equation ax=b (mod m)
        Args:
            a,b,m: input integers
        Returns: 
            print out solution in the form x=c (mod n)
    """
    d=math.gcd(a,m)

    # no solution if b is not divisible by d
    if b%d!=0:
        return "No solution!"

    # divide a,b,m by d and solve the resulting equation
    else:
        a,b,m=a//d, b//d, m//d
        a_inverse=bezout(a,m)[0]
        x=a_inverse*b % m
        return f"Solution : x = {x} (mod {m})"

# run the server locally with stdio
# npx @modelcontextprotocol/inspector uv run research_server.py
if __name__=="__main__":
    mcp.run(transport="stdio")