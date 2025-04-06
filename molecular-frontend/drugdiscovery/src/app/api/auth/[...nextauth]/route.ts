import NextAuth from "next-auth";
import GoogleProvider from "next-auth/providers/google";
import CredentialsProvider from "next-auth/providers/credentials";
import { PrismaAdapter } from "@auth/prisma-adapter";
import { PrismaClient } from "@prisma/client";
import { compare } from "bcrypt";

const prisma = new PrismaClient();

const handler = NextAuth({
  adapter: PrismaAdapter(prisma),
  providers: [
    GoogleProvider({
      clientId: process.env.GOOGLE_CLIENT_ID ?? "",
      clientSecret: process.env.GOOGLE_CLIENT_SECRET ?? "",
    }),
    CredentialsProvider({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) {
          return null;
        }

        // For now, we'll do a basic authentication without actual password comparison
        // In a real app, you would need to implement proper user lookup and password comparison
        // This is a placeholder that will allow any login for development purposes
        
        // Placeholder: In production, you would:
        // 1. Find user by email
        // 2. Compare hashed password
        // 3. Return user if valid, null if invalid
        
        // Example implementation:
        /*
        const user = await prisma.user.findUnique({
          where: { email: credentials.email }
        });
        
        if (!user || !user.password) return null;
        
        const isValid = await compare(credentials.password, user.password);
        
        if (!isValid) return null;
        
        return {
          id: user.id,
          name: user.name,
          email: user.email,
          image: user.image
        };
        */
        
        // For development, create a mock user:
        return {
          id: "mock-user-id",
          name: "Test User",
          email: credentials.email,
          image: null
        };
      }
    })
  ],
  session: {
    strategy: "jwt"
  },
  pages: {
    signIn: "/auth/signin",
    signOut: "/auth/signout",
    error: "/auth/error",
  },
  callbacks: {
    async session({ session, token }) {
      if (token && session.user) {
        session.user.id = token.sub as string;
      }
      return session;
    },
  },
  debug: process.env.NODE_ENV === "development",
});

export { handler as GET, handler as POST }; 