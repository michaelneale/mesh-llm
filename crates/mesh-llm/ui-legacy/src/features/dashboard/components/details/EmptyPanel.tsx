import { Card, CardContent } from "../../../../components/ui/card";

export function EmptyPanel({ text }: { text: string }) {
  return (
    <Card>
      <CardContent className="p-4 text-sm text-muted-foreground">
        {text}
      </CardContent>
    </Card>
  );
}
