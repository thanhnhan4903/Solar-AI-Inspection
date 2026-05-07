import React from "react";

export function Card({ className = "", children, ...props }) {
  return (
    <div
      className={`rounded-xl border bg-white shadow-sm ${className}`}
      {...props}
    >
      {children}
    </div>
  );
}

export function CardContent({ className = "", children, ...props }) {
  return (
    <div className={`p-6 pt-0 ${className}`} {...props}>
      {children}
    </div>
  );
}
